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

from streamlink.utils.parse import parse_xml
from typing import Any
from collections.abc import Mapping

period_sync_queue = queue.Queue()
rep_sync_queue = queue.Queue()

log = getLogger(__name__)

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

        # process and store plugin options before passing streams back
        for option in DASHDRM_OPTIONS:
            if option == 'decryption-key':
                if self.get_option('decryption-key'):
                    self.session.options[option] = self._process_keys()
            elif option == 'always-play-last-period':
                # jump to last period unless period is specifically set
                if not params.get('period'):
                    params['period'] = -1
            else:
                self.session.options[option] = self.get_option(option)

        return DASHStreamDRM.parse_manifest(self.session,
                                            url,
                                            **params)

    def _process_keys(self):
        """
            Parse user-supplied decryption keys.

            Keys may be supplied either as:

                key

            or

                kid:key

            Returns a list of tuples.
            Each tuple contains (kid, key), where kid is None if no KID was supplied.
            """
        keys = self.get_option('decryption-key')
        # if a colon separated key is given, assume its kid:key
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
                # key len of 21-24 may mean a base64 key was provided, so we
                # try and decode it
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
                # sanity check that it's a valid hex string
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
    '''
    Inherit and extend the FFMPEGMuxer class to pass decryption keys
    to ffmpeg

    The caller supplies a list of decryption keys corresponding to the input
    streams. Each entry is either a hexadecimal decryption key or None for an
    unencrypted stream. When a key is present, it is passed to FFMPEG using the
    -decryption_key input option.
    '''

    def __init__(self, session, *streams, **options):
        keys = options.pop("keys", None) or []
        if keys and len(keys) != len(streams):
            raise PluginError(f"Decryption key count ({len(keys)}) does not match stream count ({len(streams)}).")

        super().__init__(session, *streams, **options)
        # if a decryption key is set, we rebuild the ffmpeg command list
        # to include the key before specifying the input stream
        k = 0
        subtitles = self.session.options.get("use-subtitles")
        vid_codec_preset = None
        if session.options.get("video-codec-preset"):
            vid_codec_preset = self.session.options.get("video-codec-preset")
        # Build new ffmpeg command list
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
                self._cmd.extend(['-thread_queue_size', '4096'])
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
        #self._cmd.extend(["-report"])
        log.debug("Updated ffmpeg command %s", self._cmd)


class DASHStreamWriterDRM(DASHStreamWriter):
    reader: DASHStreamReaderDRM
    stream: DASHStreamDRM

    def fetch(self, segment: DASHSegment):
        if self.closed:
            return

        real_available_in = (segment.available_at - now()).total_seconds()
        name = segment.name
        log.debug(f"{self.reader.mime_type} segment {name}: Available in {real_available_in:.01f}s ({segment.availability})")

        if self.session.options.get("availability-grace"):
            availability_grace = float(self.session.options.get(
                                    "availability-grace"))
            segment.available_at = segment.available_at + timedelta(
                                            seconds=availability_grace)
            log.debug(f"{self.reader.mime_type} segment {name}: Adding {availability_grace} seconds to segment availability")

        if self.session.options.get("ignore-availability"):
            segment.available_at = fromtimestamp(0)
            log.debug(f"Ignoring availability timestamps. Now avallable at {segment.available_at}")

        return super().fetch(segment)


class DASHStreamWorkerDRM(DASHStreamWorker):
    reader: DASHStreamReaderDRM
    writer: DASHStreamWriterDRM
    stream: DASHStreamDRM

    def iter_segments(self):
        '''
        This is copy of iter_segments, but with DRM checks disabled,
        and slight change to limit max amount of time to wait before
        looking for segments
        '''
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
            # find the representation by ID
            representation = self.mpd.get_representation(self.reader.ident)

            # check if a new representation is available
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
                        # dont take the whole rep duration as wait time
                        # as some mpd will set a large number. we then
                        # end up staying in the sleeper loop too long
                        # and ffmpeg will timeout
                        min(representation.period.duration.total_seconds(),5)
                        if representation else 0,
                    )
                    or 5
                )

            if new_rep and not queued:
                # New rep available and no yield so we swap to the new one
                self.reader.ident = new_rep.ident
                representation = new_rep
                new_rep = None
                log.debug("New period and no yield. Swapping to next period.")
                if video_is_filtered:
                    log.debug("New period %s marked to be filtered.", self.reader.ident)
                    filtered = True
                    representation = None
            elif new_rep and queued:
                # New rep available but we had yield so we dont swap yet.
                # Set refresh to be very low since we know we actually have
                # new content in the from of new_rep
                refresh_wait = 1
                log.debug("New period and but we have yield. Not swapping to next period yet.")

            with self.sleeper(refresh_wait * back_off_factor):
                if not representation:
                    if filtered:
                        log.debug("Period is marked to be filtered.")
                        if last_segment:
                            repeated += 1
                            log.debug("Repeating last segment %s times.", repeated)
                            yield last_segment
                        else:
                            log.debug("No last segment to repeat.")
                    self.reload()
                    continue
                queued = False
                iter_segments = representation.segments(
                    sequence=self.sequence,
                    init=init,
                    # sync initial timeline generation between audio and video threads
                    timestamp=self.reader.timestamp if init else None,
                )
                for segment in iter_segments:
                    if init and not segment.init:
                        self.sequence = segment.num
                        init = False
                    last_segment = segment
                    repeated = 0
                    queued |= yield segment

                # close worker if type is not dynamic (all segments were put into writer queue)
                if self.mpd.type != "dynamic":
                    self.close()
                    return

                # Implicit end of stream

                #if self.check_queue_deadline(queued):
                #    return

                if not self.reload():
                    # use min instead of max to limit run-away backoff
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
        '''
        Check whether there are any more periods in the overall list of periods
        beyond the current period id. If so, return the index for the next period
        otherwise return 0
        '''
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

        # tell the audio streams
        audio_count = len(self.stream.audio_representations)
        if audio_count == 1 and self.stream.audio_representations == [None]:
            audio_count = 0
        if audio_count > 0:
            for _ in range(audio_count):
                period_sync_queue.put(next_period)
        # tell the subtitle streams
        subtitles_count = len(self.stream.subtitles_representations)
        if subtitles_count == 1 and self.stream.subtitles_representations == [None]:
            subtitles_count = 0
        if subtitles_count > 0:
            for _ in range(subtitles_count):
                period_sync_queue.put(next_period)
        return next_period

    def check_new_rep(self, first_rep=None):
        '''
        Check if new representation is available, if so find the matching stream
        and return with the new rep's stream object
        '''
        new_video_rep = None
        is_filtered = False
        audio_count = len(self.stream.audio_representations)
        if audio_count == 1 and self.stream.audio_representations == [None]:
            audio_count = 0
        subtitles_count = len(self.stream.subtitles_representations)
        if subtitles_count == 1 and self.stream.subtitles_representations == [None]:
            subtitles_count = 0
        log.debug("Audio stream count: %s", audio_count)
        log.debug("Subtitles stream count: %s", subtitles_count)

        if first_rep:
            is_filtered = self.rep_check_filtered(first_rep)
            log.debug("Telling non-video streams first rep and filter")
            # tell the audio streams
            if audio_count > 0:
                for _ in range(audio_count):
                    rep_sync_queue.put((new_video_rep, is_filtered))
            # tell the subtitle streams
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
            # tell the audio streams
            if rep_sync_queue.qsize() > 0:
                log.debug("Current rep_sync queue size is non-zero (%s), waiting 5 seconds to give other streams time to catch up", rep_sync_queue.qsize())
                self.sleeper(5)
                if rep_sync_queue.qsize() > 0:
                    log.debug("Current rep_sync queue size is still non-zero (%s). This means streams are out of sync.", rep_sync_queue.qsize())
            if audio_count > 0:
                for _ in range(audio_count):
                    rep_sync_queue.put((new_video_rep, is_filtered))
            # tell the subtitle streams
            if subtitles_count > 0:
                for _ in range(subtitles_count):
                    rep_sync_queue.put((new_video_rep, is_filtered))
            log.debug("Current rep_sync queue size: %s", rep_sync_queue.qsize())
        return (new_video_rep, is_filtered)

class DASHStreamWorkerDRMNonVideo(DASHStreamWorkerDRM):
    reader: DASHStreamReaderDRM
    writer: DASHStreamWriterDRM
    stream: DASHStreamDRM

    def next_period_available(self):
        '''
        Check whether there are any more periods in the overall list of periods
        beyond the current period id. If so, return the index for the next period
        otherwise return 0
        '''
        period_id = self.reader.ident[0]
        current_period_ids = [ p.id for p in self.mpd.periods ]
        current_period_idx = current_period_ids.index(period_id)

        log.debug("Current playing period: %s", current_period_idx + 1)
        log.debug("Number of periods: %s", len(current_period_ids))
        next_period = period_sync_queue.get(block=True)
        return next_period

    def check_new_rep(self, stream_type=None, first_rep=None):

        new_stream_rep = None
        is_filtered = False
        if first_rep:
            log.debug("Getting from video streams first rep and filter")
            log.debug("Current rep_sync queue size just before get: %s", rep_sync_queue.qsize())
            _, is_filtered = rep_sync_queue.get(block=True)
            return (None, is_filtered)

        log.debug("Checking new reps from video stream")
        next_period = self.next_period_available()
        if next_period:
            log.debug("Got next period %s from video stream", next_period)
            reloaded_streams = DASHStreamDRM.parse_manifest(self.session,
                                                        self.mpd.url,
                                                        next_period)
            _, a, r = self.reader.ident
            new_stream_rep = self.mpd.get_representation((next_period, a, r))
            if new_stream_rep:
                log.debug("New %s rep found. New ident: %s", stream_type, new_stream_rep.ident)
            else:
                log.debug("New period found, but can't find matching %s rep, trying to find the stream the old way", stream_type)
                reload_stream = reloaded_streams[self.stream.stream_name]
                if stream_type == "audio":
                    audio_num=int(self.name[-1])
                    new_stream_rep = reload_stream.audio_representations[audio_num]
                elif stream_type == "sub":
                    new_stream_rep = reload_stream.subtitles_representation
                if new_stream_rep:
                    log.debug("New %s representation found!", stream_type)

            log.debug("Checking video rep and filter status from video stream")
            log.debug("Current rep_sync queue size just before get: %s", rep_sync_queue.qsize())
            new_video_rep, is_filtered = rep_sync_queue.get(block=True)
            log.debug("Video stream sent video rep %s and filter status %s", new_video_rep, is_filtered)
            log.debug("New video rep: %s, isfiltered: %s", new_video_rep.ident, is_filtered)
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
        log.debug("Subtitle stream segment: %s", _)
        return _

class DASHStreamDRM(DASHStream):
    """
    Implementation of the "Dynamic Adaptive Streaming over HTTP" protocol (MPEG-DASH)
    """
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
            audio_representations[0] if audio_representations[0] else None,
            duration,
            **kwargs,
        )
        self.audio_representations = audio_representations
        self.subtitles_representations = subtitles_representations

    __shortname__ = "dashdrm"

    # FFmpeg's decryption_key input option is not supported by these
    # demuxers/containers.
    FFMPEG_UNSUPPORTED_DECRYPTION_MIME_SUBTYPES = {
        "webm",
        "x-matroska",
        "x-matroska-3d",
    }

    @staticmethod
    def parse_mpd(session, manifest: str, mpd_params: Mapping[str, Any]) -> MPD:
        node = parse_xml(manifest, ignore_ns=True)
        if session.options.get("ignore-location"):
            location = node.find('Location')
            if location is not None:
                log.warning('Found Location tag: %s', location.text)
                parent = location.getparent()
                parent.remove(location)
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
        """
        Parse a DASH manifest file and return its streams.

        :param session: Streamlink session instance
        :param url_or_manifest: URL of the manifest file or an XML manifest string
        :param period: Which MPD period to use (index number (int) or ``id`` attribute (str)) for finding representations
        :param with_video_only: Also return video-only streams, otherwise only return muxed streams
        :param with_audio_only: Also return audio-only streams, otherwise only return muxed streams
        :param kwargs: Additional keyword arguments passed to :meth:`requests.Session.request`
        """

        manifest, mpd_params = cls.fetch_manifest(session, url_or_manifest, **kwargs)

        try:
            mpd = cls.parse_mpd(session, manifest, mpd_params)
        except Exception as err:
            raise PluginError(f"Failed to parse MPD manifest: {err}") from err

        if session.options.get("presentation-delay"):
            presentation_delay = session.options.get("presentation-delay")
            mpd.suggestedPresentationDelay = timedelta(
                                                seconds=int(presentation_delay)
                                                )

        source = mpd_params.get("url", "MPD manifest")
        video: list[Representation | None] = [None] if with_audio_only else []
        audio: list[Representation | None] = [None] if with_video_only else []
        subtitles: list[Representation | None] = [None] if with_audio_only else []

        available_periods = [f"{idx}{f' (id={p.id!r})' if p.id is not None else ''}" for idx, p in enumerate(mpd.periods)]
        log.debug(f"Available DASH periods: {', '.join(available_periods)}")

        try:
            if isinstance(period, int):
                period_selection = mpd.periods[period]
            else:
                period_selection = mpd.periods_map[period]
        except LookupError:
            raise PluginError(
                f"DASH period {period!r} not found. Select a valid period by index or by id attribute value.",
            ) from None

        # Search for suitable video and audio representations
        for aset in period_selection.adaptationSets:
            aset_encrypted = bool(aset.contentProtections)
            if aset_encrypted:
                if not session.options.get("decryption-key"):
                    raise PluginError(f"{source} is protected by DRM but no key given")
                else:
                    log.debug(f"{source} is protected by DRM")
            for rep in aset.representations:
                rep_encrypted = aset_encrypted or bool(rep.contentProtections)
                if rep.contentProtections:
                    if not session.options.get("decryption-key"):
                        raise PluginError(f"{source} is protected by DRM but no key given")
                    else:
                        log.debug(f"{source} is protected by DRM")

                if rep_encrypted:
                    mime_subtype = rep.mimeType.partition("/")[2].lower()
                    if mime_subtype in cls.FFMPEG_UNSUPPORTED_DECRYPTION_MIME_SUBTYPES:
                        log.debug(
                            "Ignoring encrypted representation %s (%s): "
                            "FFmpeg does not support decryption of this container format",
                            rep.ident,
                            rep.mimeType,
                        )
                        continue

                if rep.mimeType.startswith("video"):
                    video.append(rep)
                elif rep.mimeType.startswith("audio"):  # pragma: no branch
                    audio.append(rep)
                elif (session.options.get("use-subtitles") and
                        rep.mimeType.startswith("application")):
                    subtitles.append(rep)

                if session.options.get("store-representation-kid"):
                    if rep_encrypted:
                        rep.kid = cls._get_representation_kid(session, rep)
                        log.debug(
                            "Representation %s KID=%s",
                            rep.ident,
                            rep.kid,
                        )
                    else:
                        rep.kid = None

        if not video:
            video.append(None)
        if not audio:
            audio.append(None)
        if not subtitles:
            subtitles.append(None)

        locale = session.localization
        locale_lang = locale.language
        lang = None
        available_languages = set()

        # if the locale is explicitly set, prefer that language over others
        for aud in audio:
            if aud and aud.lang:
                available_languages.add(aud.lang)
                with suppress(LookupError):
                    if locale.explicit and aud.lang and Language.get(aud.lang) == locale_lang:
                        lang = aud.lang

        if not lang:
            # filter by the first language that appears
            lang = audio[0].lang if audio[0] else None

        if session.options.get("language"):
            # if we set a specific language, try and use it for filtering
            lang = session.options.get("language")

        log.debug(
            f"Available languages for DASH audio streams: {', '.join(available_languages) or 'NONE'} (using: {lang or 'n/a'})",
        )

        if session.options.get("disable-multi-audio"):
            # if the language is given by the stream, filter out other languages that do not match
            if len(available_languages) > 1:
                audio = [a for a in audio if a and (a.lang is None or a.lang == lang)]

        ret = []
        for vid, aud in itertools.product(video, audio):
            if not vid and not aud:
                continue

            if session.options.get("disable-multi-audio"):
                # stream with single audio
                stream = DASHStreamDRM(session, mpd, vid, [aud], subtitles, **kwargs)
            else:
                # stream with all audios
                stream = DASHStreamDRM(session, mpd, vid, audio, subtitles, **kwargs)
            stream_name = []

            if vid:
                stream_name.append(f"{vid.height or vid.bandwidth_rounded:0.0f}{'p' if vid.height else 'k'}")
            if aud and len(audio) > 1:
                stream_name.append(f"a{aud.bandwidth:0.0f}k")
            ret.append(("+".join(stream_name), stream))

        # rename duplicate streams
        dict_value_list = defaultdict(list)
        for k, v in ret:
            dict_value_list[k].append(v)

        def sortby_bandwidth(dash_stream: DASHStreamDRM) -> float:
            if dash_stream.video_representation:
                return dash_stream.video_representation.bandwidth
            if session.options.get("disable-multi-audio"):
                if dash_stream.audio_representation:
                    return dash_stream.audio_representation.bandwidth
            return 0  # pragma: no cover

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

        # add stream_name to the returned streams so we can find it again
        for stream_name in ret_new:
            ret_new[stream_name].stream_name = stream_name

        return ret_new

    def _resolve_decryption_keys(self, readers):
        """
        Resolve one decryption key for each input stream.

        If user-supplied KIDs are available, representations are matched by KID.
        If KID matching cannot be completed for every stream, positional key
        assignment is used instead.

        Returns a list of keys aligned with the supplied reader objects.
        """
        in_keys = self.session.options.get("decryption-key")
        if not in_keys:
            return []

        kid_lookup = {
            kid: key
            for kid, key in in_keys
            if kid is not None
        }

        def positional_match():
            keys = [key for _, key in in_keys]
            # If only 1 key is given, then we use that also for all remaining streams
            if len(keys) == 1:
                return [keys[0]] * len(readers)
            out_keys = []
            key = 0
            for _ in readers:
                out_keys.append(keys[key])
                key += 1
                # If we had more streams than keys, start with the first audio key again
                if key == len(keys):
                    key = 1
            return out_keys

        if not kid_lookup:
            log.debug("No KIDs provided, using positional key assignment")
            return positional_match()

        rtn_keys = []

        for reader in readers:
            representation = self.mpd.get_representation(reader.ident)
            if not representation:
                log.debug("Unable to find representation, falling back to positional key assignment")
                return positional_match()
            kid = getattr(representation, "kid", None)
            if kid:
                log.debug(
                    "Representation %s KID=%s",
                    representation.ident,
                    kid,
                )
            else:
                log.debug("Representation %s missing kid attr", representation.ident)
            key = kid_lookup.get(kid)
            if key is None:
                log.debug(
                    "Unable to match representation %s by KID, falling back to positional key assignment",
                    representation.ident,
                )
                return positional_match()

            rtn_keys.append(key)

        log.debug("Successfully matched all representations by KID")
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

        if index == -1:
            return None

        if index + 28 > len(data):
            return None

        return data[index + 12:index + 28].hex()

    @classmethod
    def _get_representation_kid(
            cls,
            session: Streamlink,
            rep: Representation,
    ) -> str | None:

        # Get from MPD
        for cp in rep.contentProtections:
            if cp.default_KID:
                return cp.default_KID.replace("-", "").lower()

        for cp in rep.parent.contentProtections:
            if cp.default_KID:
                return cp.default_KID.replace("-", "").lower()

        # Fallback to init segment (only works for MP4)
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

        maps = ["0:v?", "0:a?"]
        metadata = {}

        if rep_video:
            video = DASHStreamReaderDRMVideo(self, rep_video, timestamp, name="video")
            log.debug(f"Opening DASH reader for: {rep_video.ident!r} - {rep_video.mimeType}")
            video.open()
            fds.append(video)
       
        #if rep_audio:
        #    audio = DASHStreamReaderDRM(self, rep_audio, timestamp)
        #    log.debug(f"Opening DASH reader for: {rep_audio.ident!r} - {rep_audio.mimeType}")

        next_map = 1
        if rep_audios:
            for i, rep_audio in enumerate(rep_audios):
                audio = DASHStreamReaderDRMAudio(self, rep_audio, timestamp, name="audio"+str(i))
                if not audio1:
                    audio1 = audio
                log.debug(f"Opening DASH reader for: {rep_audio.ident!r} - {rep_audio.mimeType}")
                audio.open()
                fds.append(audio)
                metadata["s:a:{0}".format(i)] = ["language={0}".format(rep_audio.lang), "title=\"{0}\"".format(rep_audio.lang)]
            maps.extend(f"{i}:a" for i in range(next_map, next_map + len(rep_audios)))
            next_map = len(rep_audios) + 1

        # only do subtitles if we have video
        if rep_subtitles and rep_subtitles[0] and rep_video:
            for _, rep_subtitle in enumerate(rep_subtitles):
                #if not rep_subtitle:
                    #break
                subtitle = DASHStreamReaderDRMSub(self, rep_subtitle, timestamp, name="subtitle"+str(_))
                log.debug(f"Opening DASH reader for: {rep_subtitle.ident!r} - {rep_subtitle.mimeType}")
                subtitle.open()
                fds.append(subtitle)
                metadata["s:s:{0}".format(_)] = ["language={0}".format(rep_subtitle.lang), "title=\"{0}\"".format(rep_subtitle.lang)]
            maps.extend(f"{_}:s" for _ in range(next_map, next_map + len(rep_subtitles)))

        keys = self._resolve_decryption_keys(fds)

        if video and audio and FFMPEGMuxerDRM.is_usable(self.session):
            return FFMPEGMuxerDRM(self.session, *fds, keys=keys, copyts=True, maps=maps, metadata=metadata).open()
        elif video:
            return video
        elif audio:
            return audio1


__plugin__ = MPEGDASHDRM
