# streamlink-plugin-dashdrm

A streamlink plugin that extends the standard streamlink dash plugin to support DRM based MPDs, as well as multi audio and multi period support.

This is a reimplementation of [streamlink-drm](https://github.com/ImAleeexx/streamlink-drm) as a plugin, so the latest streamlink base code can be used. In addition to DRM support, the plugin also reimplement the multiple audio fork done by [Shea Smith](https://github.com/SheaSmith/streamlink-drm). 

In addition, the plugin also implements multi period support. Note that the [caveats](https://github.com/streamlink/streamlink/issues/5058) around periods with differing video specifications likely causing video players to not play properly still applies, so YMMV. Lastly, the plugin currently contains code to also include subtitle streams, however ffmpeg has no support for TTML decoding at this stage, so any TTML/STPP based subtitles will not be recognised by the player even if it is included.

# Install and Use

To use this plugin, you need to utilise streamlink's plugin [sideload](https://streamlink.github.io/latest/cli/plugin-sideloading.html) capability. Download the plugin source (dashdrm.py) and either place it in your streamlink plugins sideload directory, or put in a new directory and specify the path when executing streamlink with --plugin-dir <path_of_dashdrm.py>.

To make use of the plugin, add dashdrm:// in front of the url.
```sh
Eg: streamlink --url dashdrm://http://abc.def/xyz.mpd
```

# Parameters

The plugin accepts a number of optional parameters:
<TABLE>
  <TR>
    <TH>Option</TH>
    <TH>Description</TH>
  </TR>
  <TR>
    <TD>--dashdrm-decryption-key <key in hex></TD>
    <TD>This is the decryption key to be passed to ffmpeg</TD>
  </TR>
  <TR>
    <TD>--dashdrm-presentation-delay <delay in seconds></TD>
    <TD>Override the presentation delay defaults, similar to hls-live-edge</TD>
  </TR>
  <TR>
    <TD>--dashdrm-last-period</TD>
    <TD>Jump straight to the last period, useful to skip pre-rolls</TD>
  </TR>
  <TR>
    <TD>--dashdrm-use-subtitles</TD>
    <TD>Mux in subtitle tracks that are found, however ffmpeg currently does not support TTML/STPP</TD>
  </TR>
</TABLE>
