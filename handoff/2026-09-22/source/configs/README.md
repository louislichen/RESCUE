# configs/ — frozen experiment configs, populations and prior manifests

Every file here is a recorded input of a valid run. If the recorded config hash or identity
does not match the run being reproduced, results are not comparable.

Important identities:

- four-arm config hash: `d311230d8696d388b240b08e195ec11fed350d2c3dbd8011fb94a1456d7eab97`
- decision-relevance config hash: `db8f6dfc436b2084330684ccaef1d858913b4d0f5b78ce24536e12187f75cbcb`
- TD config hash: `34bc3928f0bdb0ec9dc8379ea766880421f8be8928d8c5d83814e59244885fe5`
- training config sha256: `81dd4b1a8c2db6ffb3bd1afd6539b24d5d9a5750eebec74bc0916a1fad26413a`
- validation split sha256: `d4686c26f50cbaaf3788eef79a3a12af49a5224671e1d3b5a02195f299627ae7`
- checkpoint sha256: `6532daad4e3ae54a7e76063e6aef3fef9b59ba6ef709394bf361c30228f19ace`

Execution-host absolute paths in recorded configs are historical provenance, not portable defaults.
For a fresh run, supply local paths through the runner arguments; do not edit a frozen config and
pretend its hash is unchanged.
