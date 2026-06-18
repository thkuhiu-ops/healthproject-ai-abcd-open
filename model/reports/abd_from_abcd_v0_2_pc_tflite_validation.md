# ABD From ABCD v0.2 PC TFLite Validation

- Model path: `LOCAL_PATH_REMOVED
- NPZ path: `LOCAL_PATH_REMOVED
- Input shape: `[20, 107]`
- TFLite input details: `[{"name": "serving_default_abd_flat_input:0", "shape": [1, 107], "dtype": "float32"}]`
- TFLite output details: `[{"name": "PartitionedCall:0", "shape": [1, 3], "dtype": "float32"}]`
- Output shape: `[20, 3]`
- Sample count: `20`
- Input contains NaN/Inf: `False`
- Output contains NaN/Inf: `False`
- Predicted distribution: `{"UNCERTAIN_OR_MOTION": 12, "SIGNAL_BAD_OR_CONTACT_BAD": 8}`
- Max abs diff vs folded-source reference: `7.450580597e-08`
- Mean abs diff vs folded-source reference: `1.101846792e-08`
- Expected output CSV: `LOCAL_PATH_REMOVED

## Per-Frame Softmax

- sample `0`: prob0=`0.00169614`, prob1=`0.00019620`, prob2=`0.99810767`, pred=`2:UNCERTAIN_OR_MOTION`
- sample `1`: prob0=`0.01589950`, prob1=`0.78400534`, prob2=`0.DATE_REMOVED`, pred=`1:SIGNAL_BAD_OR_CONTACT_BAD`
- sample `2`: prob0=`0.01589950`, prob1=`0.78400534`, prob2=`0.DATE_REMOVED`, pred=`1:SIGNAL_BAD_OR_CONTACT_BAD`
- sample `3`: prob0=`0.01589950`, prob1=`0.78400534`, prob2=`0.DATE_REMOVED`, pred=`1:SIGNAL_BAD_OR_CONTACT_BAD`
- sample `4`: prob0=`0.01589950`, prob1=`0.78400534`, prob2=`0.DATE_REMOVED`, pred=`1:SIGNAL_BAD_OR_CONTACT_BAD`
- sample `5`: prob0=`0.01589950`, prob1=`0.78400534`, prob2=`0.DATE_REMOVED`, pred=`1:SIGNAL_BAD_OR_CONTACT_BAD`
- sample `6`: prob0=`0.01589950`, prob1=`0.78400534`, prob2=`0.DATE_REMOVED`, pred=`1:SIGNAL_BAD_OR_CONTACT_BAD`
- sample `7`: prob0=`0.00268230`, prob1=`0.00410192`, prob2=`0.99321586`, pred=`2:UNCERTAIN_OR_MOTION`
- sample `8`: prob0=`0.01589950`, prob1=`0.78400534`, prob2=`0.DATE_REMOVED`, pred=`1:SIGNAL_BAD_OR_CONTACT_BAD`
- sample `9`: prob0=`0.01589950`, prob1=`0.78400534`, prob2=`0.DATE_REMOVED`, pred=`1:SIGNAL_BAD_OR_CONTACT_BAD`
- sample `10`: prob0=`0.00175946`, prob1=`0.00105645`, prob2=`0.99718410`, pred=`2:UNCERTAIN_OR_MOTION`
- sample `11`: prob0=`0.00162553`, prob1=`0.00093145`, prob2=`0.99744296`, pred=`2:UNCERTAIN_OR_MOTION`
- sample `12`: prob0=`0.00090686`, prob1=`0.00012546`, prob2=`0.99896765`, pred=`2:UNCERTAIN_OR_MOTION`
- sample `13`: prob0=`0.00094008`, prob1=`0.00013270`, prob2=`0.99892730`, pred=`2:UNCERTAIN_OR_MOTION`
- sample `14`: prob0=`0.00196901`, prob1=`0.00042984`, prob2=`0.99760121`, pred=`2:UNCERTAIN_OR_MOTION`
- sample `15`: prob0=`0.00223887`, prob1=`0.00052726`, prob2=`0.99723393`, pred=`2:UNCERTAIN_OR_MOTION`
- sample `16`: prob0=`0.00223887`, prob1=`0.00052726`, prob2=`0.99723393`, pred=`2:UNCERTAIN_OR_MOTION`
- sample `17`: prob0=`0.00156910`, prob1=`0.00029959`, prob2=`0.99813133`, pred=`2:UNCERTAIN_OR_MOTION`
- sample `18`: prob0=`0.00154907`, prob1=`0.00029354`, prob2=`0.99815732`, pred=`2:UNCERTAIN_OR_MOTION`
- sample `19`: prob0=`0.00278911`, prob1=`0.01255809`, prob2=`0.98465282`, pred=`2:UNCERTAIN_OR_MOTION`

## Deployment Bundle

- NPZ: `LOCAL_PATH_REMOVED
- NPY: `LOCAL_PATH_REMOVED
- CSV: `LOCAL_PATH_REMOVED
