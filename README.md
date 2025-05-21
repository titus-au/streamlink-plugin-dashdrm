# streamlink-plugin-dashdrm

A [streamlink](https://github.com/streamlink/streamlink) plugin that extends the standard streamlink dash plugin to support DRM based MPDs, as well as multi audio and multi period support.

This is a reimplementation of [streamlink-drm](https://github.com/ImAleeexx/streamlink-drm) as a plugin, so the latest streamlink base code can be used. In addition to DRM support, the plugin also reimplement the multiple audio fork done by [Shea Smith](https://github.com/SheaSmith/streamlink-drm). 

In addition, the plugin also implements multi period support. Note that the [caveats](https://github.com/streamlink/streamlink/issues/5058) around periods still applies, such as differing video specifications likely causing video players to not play properly, or when switching to a new period too quickly thus cutting off the previous period, so YMMV. Lastly, the plugin currently contains code to also include subtitle streams, however ffmpeg has no support for TTML decoding at this stage, so any TTML/STPP based subtitles will not be recognised by the player even if it is included.

# Install and Use

To use this plugin, you need to utilise streamlink's plugin [sideload](https://streamlink.github.io/latest/cli/plugin-sideloading.html) capability. Download the plugin source by cloning the repository, or just downloading the plugin file (dashdrm.py) and either place it in your streamlink plugins sideload directory, or put in a new directory and specify the path when executing streamlink with --plugin-dir <path_of_dashdrm.py>.

To make use of the plugin, add dashdrm:// in front of the url.
```sh
streamlink --plugin-dir /path/to/dashdrm/plugin --default-stream best --url dashdrm://http://abc.def/xyz.mpd
```
Multi period support is enabled automatically, however if you wish to jump straight to the current last period when streamlink starts, you can do:
```sh
streamlink --plugin-dir /path/to/dashdrm/plugin --default-stream best --url "dashdrm://http://abc.def/xyz.mpd period=-1"
```

# Parameters

The plugin accepts a number of optional parameters:
<TABLE>
  <TR>
    <TH>Option</TH>
    <TH>Description</TH>
  </TR>
  <TR>
    <TD>--dashdrm-decryption-key &ltkey in hex&gt</TD>
    <TD>This is a comma seperated list of decryption keys to be passed to ffmpeg. If only one key is given, then all streams will use this key. If 2 keys are given, then the video stream will use the first key, and all remaining streams (eg audio streams) will use the second key. If more than 2 keys are given, then the video stream will use the first key, and the second stream will use the second key, the third stream will use the third key etc. If more streams than keys are given, keys will be looped starting with the second key. The keys need to be in hex, either just the key by itself, or in the format of kid:key (although the kid is not used)</TD>
  </TR>
  <TR>
    <TD>--dashdrm-presentation-delay &ltdelay in seconds&gt</TD>
    <TD>Override the presentation delay defaults, similar to hls-live-edge</TD>
  </TR>
  <TR>
    <TD>--dashdrm-use-subtitles</TD>
    <TD>Mux in subtitle tracks that are found, however ffmpeg currently does not support TTML/STPP</TD>
  </TR>
</TABLE>

# Disclaimer

<LI>Use of this code to decrypt DRM is purely for academic purposes. You should not use this code for any illegal purposes and I take no responsibility for your actions</LI>
<LI>This code has not been widely tested, so consider it alpha software</LI>

