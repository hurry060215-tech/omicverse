# Explicit analytics opt-out

Set `OMICVERSE_DISABLE_ANALYTICS=1` **before importing OmicVerse** to disable
OmicVerse's analytics senders and the automatic settings-initialization event:

```python
import os
os.environ['OMICVERSE_DISABLE_ANALYTICS'] = '1'
import omicverse as ov
```

Values `true`, `yes`, and `on` are also accepted (case-insensitive). This switch
returns before analytics imports at settings initialization and before machine
fingerprinting, IP-geolocation lookup or HTTP in the senders themselves.
It does not redirect traffic to another endpoint or imply consent to telemetry.
Default behavior when unset is unchanged. This is an analytics-specific privacy
control, not a firewall: explicitly requested dataset downloads and other
network features are not disabled.

Offline validation tests load only source functions with collection and
requests patched to fail. They verify that all public senders and the settings
constructor perform zero analytics collection/network calls under the opt-out.
