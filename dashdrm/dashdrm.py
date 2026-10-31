from __future__ import annotations

import re
import itertools
import base64
import queue
from collections import defaultdict
from contextlib import suppress
from typing import List, Self
from datetime import timedelta

from streamlink.exceptions import PluginError, FatalPluginError
from streamlink.plugin import pluginmatcher, pluginargument
from streamlink.plugin.plugin import (HIGH_PRIORITY,
                                        parse_params,
                                        stream_weight)
from streamlink.stream.dash import (DASHStream,
                                    DASHStreamWorker,
                                    DASHStreamWriter,
                                    DASHStreamReader)
from streamlink.stream.dash.manifest import MPD, Representation
from streamlink.stream.dash.segment import DASHSegment
from streamlink.stream.ffmpegmux import FFMPEGMuxer
from streamlink.utils.url import update_scheme
from streamlink.session import Streamlink
from streamlink.utils.l10n import Language
from streamlink.utils.times import fromtimestamp, now
from streamlink.logger import getLogger
from streamlink.plugins.dash import MPEGDASH
from requests.adapters import HTTPAdapter

from streamlink.utils.parse import parse_xml
from typing import Any
from collections.abc import Mapping

# Attempt to load lxml for high-speed C-based XML parsing
try:
    from lxml import etree as lxml_etree
    HAS_LXML = True
except ImportError:
    HAS_LXML = False

period_sync_queue = queue.Queue()
rep_sync_queue = queue.Queue()

log = getLogger(__name__)
# Demote spammy logs to trace to save I/O overhead
log_trace = getattr(log, "trace", log.debug)

DASHDRM_OPTIONS = [
    "decryption-key",
    "presentation-delay",
    "use-subtitles",
    "ignore-location",
    "ignore-availability",
    "availability-grace",
    "always-play-last-period",
    "video-codec-preset",
    "video-timescale",
    "disable-multi-audio",
    "language",
]

@pluginmatcher(
    priority=HIGH_PRIORITY,
    pattern=re.compile(r"dashdrm://(?P<url>\S+)(?:\s(?P<params>.+))?$"),
)
@pluginargument(
    "decryption-key",
    type="comma_list",
    help="Decryption key(s) to be passed to ffmpeg. Keys may be supplied as"
    " 'key' or 'kid:key'. When KIDs are supplied, keys are matched to"
    " streams automatically, otherwise positional matching is used."
)
@pluginargument(
    "presentation-delay",
    help="Override presentation delay value (in seconds). Similar to"
    " --hls-live-edge."
)
@pluginargument(
    "use-subtitles",
    action="store_true",
    help="Experiment with subtitles."
)
@pluginargument(
    "ignore-location",
    action="store_true",
    help="Workaround to ignore Location tags that is not compliant resulting"
    " in wrong segment URLs."
)
@pluginargument(
    "ignore-availability",
    action="store_true",
    help="Workaround to ignore segment availability times where server has"
    " wrong availability segment times (eg times that are in the future)."
    " Use this with extreme caution as you are ignoring what the server is"
    " telling you in relation to when a segment is available, thus you can"
    " overwhelm a server with requests for segments that it already told"
    " you is not (yet) available."
)
@pluginargument(
    "availability-grace",
    help="Workaround to delay getting segments even though the segment"
    " availability time has been reached, as the segments are not"
    " actually avaiable yet (resulting in 403/404 errors) possibly due to"
    " mismatched server clock. A negative grace value can also be given,"
    " which can cover cases where the availability times are in the future."
)
@pluginargument(
    "always-play-last-period",
    action="store_true",
    help="Always jump to the last period, even when multiple new periods are"
    " found."
)
@pluginargument(
    "video-codec-preset",
    help="Use this to specific preset for transcode. This option is meant to be"
    " paired with --ffmpeg-video-transcode. eg transcoding to libx264 with a"
    " specific preset (eg ultrafast)"
)
@pluginargument(
    "video-timescale",
    help="Use this to only display videos with the specifie timescale (eg 50000)"
    " filtering out any other. This can be used in multi-period mpds to display"
    " the main content and filter out other content (eg ads) that has incompatible"
    " timescale which can cause issues with client/players. Warning: Experimental"
)
@pluginargument(
    "disable-multi-audio",
    action="store_true",
    help="Restore standard dash plugin function of single audio stream."
)
@pluginargument(
    "language",
    help="Allow filtering of audio to a specific language code."
)
class MPEGDASHDRM(MPEGDASH):

    def _get_streams(self):
        data = self.match.groupdict()
        url = update_scheme("https://", str(data.get("url", "")), force=False)
        params = parse_params(data.get("params"))
        log.debug(f"URL={url}; params={params}")

        # Optimize HTTP pool sizes to ensure concurrent threads don't bottleneck on socket creation
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.session.http.mount("https://", adapter)
        self.session.http.mount("http://", adapter)

        for option in DASHDRM_OPTIONS:
            if option == 'decryption-key':
                if self.get_option('decryption-key'):
                    self.session.options[option] = self._process_keys()
            elif option == 'always-play-last-period':
                if not params.get('period'):
                    params['period'] = -1
            else:
                self.session.options[option] = self.get_option(option)

        return DASHStreamDRM.parse_manifest(self.session, url, **params)

    def _process_keys(self):
        keys = self.get_option('decryption-key')
        return_keys = []
        has_kid = False
        for k in keys:
            kid = None
            parts = k.split(':', 1)
            if len(parts) == 2:
                kid, key = parts
                has_kid = True
                kid = kid.replace("-", "").lower()
                log.debug('Decryption key %s has KID %s', key, kid)
            else:
                key = parts[-1]
            key_len = len(key)
            log.debug('Decryption Key %s has %s digits', key, key_len)
            if key_len in (21, 22, 23, 24):
                log.debug("Decryption key length is too short to be hex and looks like it might be base64, so we'll try and decode it..")
                b64_string = key
                padding = 4 - (len(b64_string) % 4)
                b64_string = b64_string + ("=" * padding)
                b64_key = base64.urlsafe_b64decode(b64_string).hex()
                if b64_key:
                    key = b64_key
                    key_len = len(b64_key)
                    log.debug('Decryption Key (post base64 decode) is %s and has %s digits', key, key_len)
            if key_len == 32:
                try:
                    int(key, 16)
                except ValueError as err:
                    raise FatalPluginError("Expecting 128bit key in 32 hex digits, but the key contains invalid hex.")
            elif key_len != 32:
                raise FatalPluginError("Expecting 128bit key in 32 hex digits.")
            return_keys.append((kid, key))
        self.session.options["store-representation-kid"] = has_kid
        return return_keys


class FFMPEGMuxerDRM(FFMPEGMuxer):
    def __init__(self, session, *streams, **options):
        keys = options.pop("keys", None) or []
        if keys and len(keys) != len(streams):
            raise PluginError(f"Decryption key count ({len(keys)}) does not match stream count ({len(streams)}).")

        super().__init__(session, *streams, **options)
        k = 0
        subtitles = self.session.options.get("use-subtitles")
        vid_codec_preset = None
        if session.options.get("video-codec-preset"):
            vid_codec_preset = self.session.options.get("video-codec-preset")
        
        old_cmd = self._cmd.copy()
        self._cmd = []

        while len(old_cmd) > 0:
            cmd = old_cmd.pop(0)
            if cmd == "-i":
                _ = old_cmd.pop(0)
                if keys:
                    key = keys[k]
                    if key is not None:
                        self._cmd.extend(["-decryption_key", key])
                    k += 1
                
                # IPTV Optimization: Maximize queue sizes and enforce fast start
                self._cmd.extend(['-thread_queue_size', '8192'])
                self._cmd.extend(['-analyzeduration', '1000000', '-probesize', '1000000'])
                self._cmd.extend([cmd, _])
            elif subtitles and cmd == "-c:a":
                _ = old_cmd.pop(0)
                self._cmd.extend([cmd, _])
                self._cmd.extend(["-c:s", "copy"])
            elif vid_codec_preset and cmd == "-c:v":
                _ = old_cmd.pop(0)
                self._cmd.extend([cmd, _])
                self._cmd.extend(["-preset:v", vid_codec_preset])
            else:
                self._cmd.append(cmd)
        log.debug("Updated ffmpeg command %s", self._cmd)


class DASHStreamWriterDRM(DASHStreamWriter):
    reader: DASHStreamReaderDRM
    stream: DASHStreamDRM

    def fetch(self, segment: DASHSegment):
        if self.closed:
            return

        real_available_in = (segment.available_at - now()).total_seconds()
        name = segment.name
        log_trace(f"{self.reader.mime_type} segment {name}: Available in {real_available_in:.01f}s ({segment.availability})")

        if self.session.options.get("availability-grace"):
            availability_grace = float(self.session.options.get("availability-grace"))
            segment.available_at = segment.available_at + timedelta(seconds=availability_grace)
            log_trace(f"{self.reader.mime_type} segment {name}: Adding {availability_grace} seconds to segment availability")

        if self.session.options.get("ignore-availability"):
            segment.available_at = fromtimestamp(0)
            log_trace(f"Ignoring availability timestamps. Now avallable at {segment.available_at}")

        return super().fetch(segment)


class DASHStreamWorkerDRM(DASHStreamWorker):
    reader: DASHStreamReaderDRM
    writer: DASHStreamWriterDRM
    stream: DASHStreamDRM

    def iter_segments(self):
        init = True
        back_off_factor = 1
        new_rep = None
        queued = False
        last_segment = None
        first = True
        filtered = False
        video_is_filtered = False
        repeated = 0
        while not self.closed:
            representation = self.mpd.get_representation(self.reader.ident)

            if not first and not new_rep:
                new_rep, video_is_filtered = self.check_new_rep()
            elif first:
                _, filtered = self.check_new_rep(representation)
                if filtered:
                    representation = None
                log.debug("First run. Video filter %s.", filtered)

            first = False
            if self.mpd.type == "static":
                refresh_wait = 5
            else:
                refresh_wait = (
                    max(
                        self.mpd.minimumUpdatePeriod.total_seconds(),
                        min(representation.period.duration.total_seconds(),5) if representation else 0,
                    ) or 5
                )

            if new_rep and not queued:
                self.reader.ident = new_rep.ident
                representation = new_rep
                new_rep = None
                log.debug("New period and no yield. Swapping to next period.")
                if video_is_filtered:
                    log.debug("New period %s marked to be filtered.", self.reader.ident)
                    filtered = True
                    representation = None
            elif new_rep and queued:
                refresh_wait = 1
                log.debug("New period and but we have yield. Not swapping to next period yet.")

            with self.sleeper(refresh_wait * back_off_factor):
                if not representation:
                    if filtered:
                        log_trace("Period is marked to be filtered.")
                        if last_segment:
                            repeated += 1
                            log_trace(f"Repeating last segment {repeated} times.")
                            yield last_segment
                        else:
                            log.debug("No last segment to repeat.")
                    self.reload()
                    continue
                queued = False
                iter_segments = representation.segments(
                    sequence=self.sequence,
                    init=init,
                    timestamp=self.reader.timestamp if init else None,
                )
                for segment in iter_segments:
                    if init and not segment.init:
                        self.sequence = segment.num
                        init = False
                    last_segment = segment
                    repeated = 0
                    queued |= yield segment

                if self.mpd.type != "dynamic":
                    self.close()
                    return

                if not self.reload():
                    back_off_factor = min(back_off_factor * 1.3, 10.0)
                else:
                    back_off_factor = 1


class DASHStreamWorkerDRMVideo(DASHStreamWorkerDRM):
    reader: DASHStreamReaderDRMVideo
    writer: DASHStreamWriterDRM
    stream: DASHStreamDRM

    def rep_check_filtered(self, first_rep=None):
        if not first_rep:
            representation = self.mpd.get_representation(self.reader.ident)
        else:
            representation = first_rep
        if representation.segmentTemplate:
            current_timescale = representation.segmentTemplate.timescale
        elif representation.segmentList:
            current_timescale = representation.segmentList.timescale
        else:
            current_timescale = None
        log.debug("Current rep (%s) timescale: %s", self.reader.mime_type, current_timescale)

        is_filtered = False
        if self.session.options.get("video-timescale"):
            wanted_timescale = int(self.session.options.get("video-timescale"))
            if current_timescale and (current_timescale != wanted_timescale):
                is_filtered = True
        return is_filtered

    def next_period_available(self):
        period_id = self.reader.ident[0]
        current_period_ids = [ p.id for p in self.mpd.periods ]
        current_period_idx = current_period_ids.index(period_id)

        log.debug("Current playing period: %s", current_period_idx + 1)
        log.debug("Number of periods: %s", len(current_period_ids))

        next_period = 0
        if len(current_period_ids) > current_period_idx + 1:
            if self.session.options.get("always-play-last-period"):
                next_period = current_period_ids[-1]
            else:
                next_period = current_period_ids[current_period_idx + 1]

        audio_count = len(self.stream.audio_representations)
        if audio_count == 1 and self.stream.audio_representations == [None]:
            audio_count = 0
        if audio_count > 0:
            for _ in range(audio_count):
                period_sync_queue.put(next_period)
                
        subtitles_count = len(self.stream.subtitles_representations)
        if subtitles_count == 1 and self.stream.subtitles_representations == [None]:
            subtitles_count = 0
        if subtitles_count > 0:
            for _ in range(subtitles_count):
                period_sync_queue.put(next_period)
        return next_period

    def check_new_rep(self, first_rep=None):
        new_video_rep = None
        is_filtered = False
        audio_count = len(self.stream.audio_representations)
        if audio_count == 1 and self.stream.audio_representations == [None]:
            audio_count = 0
        subtitles_count = len(self.stream.subtitles_representations)
        if subtitles_count == 1 and self.stream.subtitles_representations == [None]:
            subtitles_count = 0

        if first_rep:
            is_filtered = self.rep_check_filtered(first_rep)
            log.debug("Telling non-video streams first rep and filter")
            if audio_count > 0:
                for _ in range(audio_count):
                    rep_sync_queue.put((new_video_rep, is_filtered))
            if subtitles_count > 0:
                for _ in range(subtitles_count):
                    rep_sync_queue.put((new_video_rep, is_filtered))
            log.debug("Current rep_sync queue size: %s", rep_sync_queue.qsize())
            return (None, is_filtered)

        log.debug("Checking for new period and representations in video stream")
        next_period = self.next_period_available()
        if next_period:
            reloaded_streams = DASHStreamDRM.parse_manifest(self.session,
                                                        self.mpd.url,
                                                        next_period)
            p, a, r = self.reader.ident
            new_video_rep = self.mpd.get_representation((next_period,a,r))
            if new_video_rep:
                log.debug("New video rep found. New ident: %s", new_video_rep.ident)
            else:
                log.debug("New period found, but can't find matching video rep, trying to find the stream the old way")
                reload_stream = reloaded_streams[self.stream.stream_name]
                new_video_rep = reload_stream.video_representation
                if new_video_rep:
                    log.debug("New video representation found!")

            is_filtered = self.rep_check_filtered(new_video_rep)

            log.debug("Telling non-video streams new rep and filter")
            if rep_sync_queue.qsize() > 0:
                log.debug("Current rep_sync queue size is non-zero (%s), waiting 5 seconds to give other streams time to catch up", rep_sync_queue.qsize())
                self.sleeper(5)
            if audio_count > 0:
                for _ in range(audio_count):
                    rep_sync_queue.put((new_video_rep, is_filtered))
            if subtitles_count > 0:
                for _ in range(subtitles_count):
                    rep_sync_queue.put((new_video_rep, is_filtered))
        return (new_video_rep, is_filtered)


class DASHStreamWorkerDRMNonVideo(DASHStreamWorkerDRM):
    reader: DASHStreamReaderDRM
    writer: DASHStreamWriterDRM
    stream: DASHStreamDRM

    def next_period_available(self):
        period_id = self.reader.ident[0]
        current_period_ids = [ p.id for p in self.mpd.periods ]
        current_period_idx = current_period_ids.index(period_id)

        try:
            next_period = period_sync_queue.get(block=True, timeout=15)
            return next_period
        except queue.Empty:
            log.warning("Timeout waiting for next period from period_sync_queue")
            return 0

    def check_new_rep(self, stream_type=None, first_rep=None):
        new_stream_rep = None
        is_filtered = False
        if first_rep:
            try:
                _, is_filtered = rep_sync_queue.get(block=True, timeout=15)
            except queue.Empty:
                log.warning("Timeout waiting for first rep filter state from rep_sync_queue")
                is_filtered = False
            return (None, is_filtered)

        next_period = self.next_period_available()
        if next_period:
            reloaded_streams = DASHStreamDRM.parse_manifest(self.session,
                                                        self.mpd.url,
                                                        next_period)
            _, a, r = self.reader.ident
            new_stream_rep = self.mpd.get_representation((next_period, a, r))
            if not new_stream_rep:
                reload_stream = reloaded_streams[self.stream.stream_name]
                if stream_type == "audio":
                    audio_num=int(self.name[-1])
                    new_stream_rep = reload_stream.audio_representations[audio_num]
                elif stream_type == "sub":
                    new_stream_rep = reload_stream.subtitles_representation

            try:
                new_video_rep, is_filtered = rep_sync_queue.get(block=True, timeout=15)
            except queue.Empty:
                log.warning("Timeout waiting for new video rep state from rep_sync_queue")
                is_filtered = False
            return (new_stream_rep, is_filtered)
        return (None, False)

class DASHStreamWorkerDRMAudio(DASHStreamWorkerDRMNonVideo):
    reader: DASHStreamReaderDRMAudio
    writer: DASHStreamWriterDRM
    stream: DASHStreamDRM

    def check_new_rep(self, first_rep=None):
        return super().check_new_rep(stream_type="audio", first_rep=first_rep)

class DASHStreamWorkerDRMSub(DASHStreamWorkerDRMNonVideo):
    reader: DASHStreamReaderDRMSub
    writer: DASHStreamWriterDRM
    stream: DASHStreamDRM

    def check_new_rep(self, first_rep=None):
        return super().check_new_rep(stream_type="sub", first_rep=first_rep)

class DASHStreamReaderDRM(DASHStreamReader):
    __worker__ = DASHStreamWorkerDRM
    __writer__ = DASHStreamWriterDRM

    worker: DASHStreamWorkerDRM
    writer: DASHStreamWriterDRM
    stream: DASHStreamDRM

class DASHStreamReaderDRMVideo(DASHStreamReaderDRM):
    __worker__ = DASHStreamWorkerDRMVideo
    __writer__ = DASHStreamWriterDRM

    worker: DASHStreamWorkerDRMVideo
    writer: DASHStreamWriterDRM
    stream: DASHStreamDRM

class DASHStreamReaderDRMAudio(DASHStreamReaderDRM):
    __worker__ = DASHStreamWorkerDRMAudio
    __writer__ = DASHStreamWriterDRM

    worker: DASHStreamWorkerDRMAudio
    writer: DASHStreamWriterDRM
    stream: DASHStreamDRM

class DASHStreamReaderDRMSub(DASHStreamReaderDRM):
    __worker__ = DASHStreamWorkerDRMSub
    __writer__ = DASHStreamWriterDRM

    worker: DASHStreamWorkerDRMSub
    writer: DASHStreamWriterDRM
    stream: DASHStreamDRM

    def read(self, size: int) -> bytes:
        _ = self.buffer.read(
            size,
            block=self.writer.is_alive(),
            timeout=self.timeout,
        )
        return _

class DASHStreamDRM(DASHStream):
    def __init__(
        self,
        session: Streamlink,
        mpd: MPD,
        video_representation: Representation | None = None,
        audio_representations: List[Representation] | None = None,
        subtitles_representations: List[Representation] | None = None,
        duration: float | None = None,
        **kwargs,
    ):
        super().__init__(
            session,
            mpd,
            video_representation,
            audio_representations[0] if audio_representations and audio_representations[0] else None,
            duration,
            **kwargs,
        )
        self.audio_representations = audio_representations
        self.subtitles_representations = subtitles_representations

    __shortname__ = "dashdrm"

    @classmethod
    def fetch_manifest(cls, session: Streamlink, url_or_manifest: str, **kwargs) -> tuple[str, dict]:
        if not url_or_manifest.startswith("http"):
            return url_or_manifest, {"url": "MPD manifest"}

        headers = kwargs.pop("headers", {})
        cache_headers = getattr(session, "_dashdrm_cache_headers", {})
        
        if url_or_manifest in cache_headers:
            headers.update(cache_headers[url_or_manifest])
        
        kwargs["headers"] = headers
        kwargs["acceptable_status"] = [200, 304]
        
        res = session.http.get(url_or_manifest, **kwargs)
        
        if res.status_code == 304:
            log_trace("MPD not modified (HTTP 304), returning cached manifest.")
            cached_manifest = getattr(session, "_dashdrm_cached_manifests", {}).get(url_or_manifest)
            if cached_manifest:
                # FIXED: Included base_url so segment relative paths resolve properly
                return cached_manifest, {"url": res.url, "base_url": res.url}

        new_cache_headers = {}
        if "ETag" in res.headers:
            new_cache_headers["If-None-Match"] = res.headers["ETag"]
        if "Last-Modified" in res.headers:
            new_cache_headers["If-Modified-Since"] = res.headers["Last-Modified"]
            
        if new_cache_headers:
            if not hasattr(session, "_dashdrm_cache_headers"):
                session._dashdrm_cache_headers = {}
                session._dashdrm_cached_manifests = {}
                
            if len(session._dashdrm_cache_headers) > 10:
                session._dashdrm_cache_headers.clear()
                session._dashdrm_cached_manifests.clear()

            session._dashdrm_cache_headers[url_or_manifest] = new_cache_headers
            session._dashdrm_cached_manifests[url_or_manifest] = res.text
            
        # FIXED: Included base_url here as well
        return res.text, {"url": res.url, "base_url": res.url}

    @staticmethod
    def parse_mpd(session, manifest: str, mpd_params: Mapping[str, Any]) -> MPD:
        node = None
        
        if HAS_LXML:
            try:
                manifest_clean = re.sub(r'(?i)\sxmlns="[^"]+"', '', manifest, count=1)
                node = lxml_etree.fromstring(manifest_clean.encode("utf-8"))
                
                for elem in node.iter():
                    if isinstance(elem.tag, str) and '}' in elem.tag:
                        elem.tag = elem.tag.split('}', 1)[1]
                        
                log_trace("MPD parsed successfully using fast lxml engine")
            except Exception as e:
                log.debug(f"lxml parsing failed, falling back to standard parser: {e}")
                node = None

        if node is None:
            node = parse_xml(manifest, ignore_ns=True)

        if session.options.get("ignore-location"):
            location = node.find('Location')
            if location is not None:
                log.warning('Found Location tag: %s', location.text)
                if HAS_LXML and node is not None and hasattr(location, "getparent"):
                    location.getparent().remove(location)
                else:
                    if location in node:
                        node.remove(location)

        return MPD(node, **mpd_params)

    @classmethod
    def parse_manifest(
        cls,
        session: Streamlink,
        url_or_manifest: str,
        period: int | str = 0,
        with_video_only: bool = False,
        with_audio_only: bool = False,
        **kwargs,
    ) -> dict[str, DASHStreamDRM]:
        manifest, mpd_params = cls.fetch_manifest(session, url_or_manifest, **kwargs)

        try:
            mpd = cls.parse_mpd(session, manifest, mpd_params)
        except Exception as err:
            raise PluginError(f"Failed to parse MPD manifest: {err}") from err

        if session.options.get("presentation-delay"):
            presentation_delay = session.options.get("presentation-delay")
            mpd.suggestedPresentationDelay = timedelta(seconds=int(presentation_delay))

        source = mpd_params.get("url", "MPD manifest")
        video: list[Representation | None] = [None] if with_audio_only else []
        audio: list[Representation | None] = [None] if with_video_only else []
        subtitles: list[Representation | None] = [None] if with_audio_only else []

        try:
            if isinstance(period, int):
                period_selection = mpd.periods[period]
            else:
                period_selection = mpd.periods_map[period]
        except LookupError:
            raise PluginError(f"DASH period {period!r} not found.") from None

        for aset in period_selection.adaptationSets:
            if aset.contentProtections and not session.options.get("decryption-key"):
                raise PluginError(f"{source} is protected by DRM but no key given")
            for rep in aset.representations:
                if rep.contentProtections and not session.options.get("decryption-key"):
                    raise PluginError(f"{source} is protected by DRM but no key given")
                if rep.mimeType.startswith("video"):
                    video.append(rep)
                elif rep.mimeType.startswith("audio"):
                    audio.append(rep)
                elif (session.options.get("use-subtitles") and rep.mimeType.startswith("application")):
                    subtitles.append(rep)

                if session.options.get("store-representation-kid"):
                    rep.kid = cls._get_representation_kid(session, rep)

        if not video: video.append(None)
        if not audio: audio.append(None)
        if not subtitles: subtitles.append(None)

        locale = session.localization
        locale_lang = locale.language
        lang = None
        available_languages = set()

        for aud in audio:
            if aud and aud.lang:
                available_languages.add(aud.lang)
                with suppress(LookupError):
                    if locale.explicit and aud.lang and Language.get(aud.lang) == locale_lang:
                        lang = aud.lang

        if not lang:
            lang = audio[0].lang if audio[0] else None

        if session.options.get("language"):
            lang = session.options.get("language")

        if session.options.get("disable-multi-audio"):
            if len(available_languages) > 1:
                audio = [a for a in audio if a and (a.lang is None or a.lang == lang)]

        ret = []
        for vid, aud in itertools.product(video, audio):
            if not vid and not aud:
                continue

            if session.options.get("disable-multi-audio"):
                stream = DASHStreamDRM(session, mpd, vid, [aud], subtitles, **kwargs)
            else:
                stream = DASHStreamDRM(session, mpd, vid, audio, subtitles, **kwargs)
            stream_name = []

            if vid:
                stream_name.append(f"{vid.height or vid.bandwidth_rounded:0.0f}{'p' if vid.height else 'k'}")
            if aud and len(audio) > 1:
                stream_name.append(f"a{aud.bandwidth:0.0f}k")
            ret.append(("+".join(stream_name), stream))

        dict_value_list = defaultdict(list)
        for k, v in ret:
            dict_value_list[k].append(v)

        def sortby_bandwidth(dash_stream: DASHStreamDRM) -> float:
            if dash_stream.video_representation:
                return dash_stream.video_representation.bandwidth
            if session.options.get("disable-multi-audio"):
                if dash_stream.audio_representation:
                    return dash_stream.audio_representation.bandwidth
            return 0 

        ret_new = {}
        for q in dict_value_list:
            items = dict_value_list[q]
            with suppress(AttributeError):
                items = sorted(items, key=sortby_bandwidth, reverse=True)

            for n in range(len(items)):
                if n == 0:
                    ret_new[q] = items[n]
                elif n == 1:
                    ret_new[f"{q}_alt"] = items[n]
                else:
                    ret_new[f"{q}_alt{n}"] = items[n]

        for stream_name in ret_new:
            ret_new[stream_name].stream_name = stream_name

        return ret_new

    def _resolve_decryption_keys(self, readers):
        in_keys = self.session.options.get("decryption-key")
        if not in_keys:
            return []

        kid_lookup = {kid: key for kid, key in in_keys if kid is not None}

        def positional_match():
            keys = [key for _, key in in_keys]
            if len(keys) == 1:
                return [keys[0]] * len(readers)
            out_keys = []
            key = 0
            for _ in readers:
                out_keys.append(keys[key])
                key += 1
                if key == len(keys):
                    key = 1
            return out_keys

        if not kid_lookup:
            return positional_match()

        rtn_keys = []
        for reader in readers:
            representation = self.mpd.get_representation(reader.ident)
            if not representation:
                return positional_match()
            kid = getattr(representation, "kid", None)
            key = kid_lookup.get(kid)
            if key is None:
                return positional_match()
            rtn_keys.append(key)
        return rtn_keys

    @staticmethod
    def _get_init_segment(session: Streamlink, rep: Representation) -> bytes | None:
        segments = rep.segments(init=True)
        try:
            segment = next(segments)
        except StopIteration:
            return None

        if not segment.init:
            return None

        request = session.http.get(segment.uri)
        if segment.byterange:
            start, length = segment.byterange
            end = start + length - 1 if length else ""
            request = session.http.get(
                segment.uri,
                headers={"Range": f"bytes={start}-{end}"},
            )
        return request.content

    @staticmethod
    def _extract_kid(data: bytes) -> str | None:
        tenc = b"tenc"
        index = data.find(tenc)
        if index == -1 or index + 28 > len(data):
            return None
        return data[index + 12:index + 28].hex()

    @classmethod
    def _get_representation_kid(cls, session: Streamlink, rep: Representation) -> str | None:
        for cp in rep.contentProtections:
            if cp.default_KID:
                return cp.default_KID.replace("-", "").lower()
        for cp in rep.parent.contentProtections:
            if cp.default_KID:
                return cp.default_KID.replace("-", "").lower()

        init = cls._get_init_segment(session, rep)
        if init is None:
            return None
        return cls._extract_kid(init)

    def open(self):
        video, audio, audio1 = None, None, None
        rep_video = self.video_representation
        rep_audios = self.audio_representations
        rep_subtitles = self.subtitles_representations

        timestamp = now()
        fds = []

        maps = ["0:v?"] 
        metadata = {}

        if rep_video:
            video = DASHStreamReaderDRMVideo(self, rep_video, timestamp, name="video")
            video.open()
            fds.append(video)
       
        if not rep_audios or not rep_audios[0]:
            maps.append("0:a?")

        next_map = 1
        if rep_audios and rep_audios[0]:
            for i, rep_audio in enumerate(rep_audios):
                if not rep_audio:
                    continue
                audio = DASHStreamReaderDRMAudio(self, rep_audio, timestamp, name="audio"+str(i))
                if not audio1:
                    audio1 = audio
                audio.open()
                fds.append(audio)
                
                lang = rep_audio.lang if rep_audio.lang else "und"
                metadata["s:a:{0}".format(i)] = ["language={0}".format(lang), "title={0}".format(lang)]
                
            maps.extend(f"{i}:a" for i in range(next_map, next_map + len(rep_audios)))
            next_map = len(rep_audios) + 1

        if rep_subtitles and rep_subtitles[0] and rep_video:
            for _, rep_subtitle in enumerate(rep_subtitles):
                if not rep_subtitle:
                    continue
                subtitle = DASHStreamReaderDRMSub(self, rep_subtitle, timestamp, name="subtitle"+str(_))
                subtitle.open()
                fds.append(subtitle)
                
                sub_lang = rep_subtitle.lang if rep_subtitle.lang else "und"
                metadata["s:s:{0}".format(_)] = ["language={0}".format(sub_lang), "title={0}".format(sub_lang)]
                
            maps.extend(f"{_}:s" for _ in range(next_map, next_map + len(rep_subtitles)))

        keys = self._resolve_decryption_keys(fds)

        if video and audio and FFMPEGMuxerDRM.is_usable(self.session):
            return FFMPEGMuxerDRM(self.session, *fds, keys=keys, copyts=True, maps=maps, metadata=metadata).open()
        elif video:
            return video
        elif audio:
            return audio1

__plugin__ = MPEGDASHDRM
