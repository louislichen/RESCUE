# PARTIAL TRAINING-ONLY FEASIBILITY PILOT



This is a 4K-only technical/feasibility experiment, not completion of the 4K+8K study.

No development or confirmation documents were opened. Native V is unchanged; the pilot evaluates module3 only.



Fit: historical training docs0..15 (64 fixed-arm calibrations,96 SGD requests,2 fixed passes). Internal monitor: historical training docs16..23, 8/8 complete paired document units.

Monitor requests: 336/336. Incomplete document units excluded in full: []. Missing cells are saved in summary.json. Full registered pilot phase analyzed.

One predeclared training seed; no epoch/seed selection. Final parameters are frozen, but stochastic softmax action RNG progresses and is replayed. Three action streams are not three independently trained models.

Each fit document/fixed arm has only1 calibration timing for rho/baseline selection. Training runs in mode blocks (32 updates each), one seed, so time drift is a limitation. Monitor randomizes the7 policy order for each document/stream/repeat on the registered physical GPU. Request-boundary process checks establish only this GPU's observed process isolation, not whole-machine isolation.

Primary comparator: fixed8_native, chosen only on fit16 equal-document mean request TPS (ties smaller budget). Rho and action-independent document baselines also use fit Native8 only. All four fixed comparisons are retained; monitor never selects a winner.

Each stream first takes the median of2 timing repeats, then the3 stream values are averaged within each document. Paired bootstrap resamples8 documents (or the explicitly partial completed subset),10000 draws, seed42, linear2.5/97.5 percentiles. These intervals are conditional on this fitted model and observed action streams, not a training-seed confidence interval. Stream SD/MCSE is descriptive only.



CSGB acceptance paired difference: mean=0.3012961896061658,95%CI=[0.16836167372744082, 0.4070442448479723],n=8 documents (TPS units percent; acceptance units tokens/scored round).

CSGB tps paired difference: mean=-2.345947968525769,95%CI=[-4.558755295151415, -0.7904468919575567],n=8 documents (TPS units percent; acceptance units tokens/scored round).



Pilot continuation gate: {"evaluated": true, "passed": false, "csgb_vs_fit_best_fixed": {"acceptance": {"mean": 0.3012961896061658, "ci95": [0.16836167372744082, 0.4070442448479723], "documents": 8, "per_document": {"16": -0.10806754221388326, "17": 0.24322061191626432, "18": 0.19185281677682875, "19": 0.37500844309354986, "20": 0.4305071622144794, "21": 0.4556112463089206, "22": 0.44607589888607935, "23": 0.376160879867087}}, "tps": {"mean": -2.345947968525769, "ci95": [-4.558755295151415, -0.7904468919575567], "documents": 8, "per_document": {"16": -9.353318374037812, "17": -3.964407825502203, "18": -0.5281112111930386, "19": -1.4228687316923283, "20": -0.3456063758012773, "21": -1.0753085165281728, "22": -0.4760669238774162, "23": -1.6018957895739039}}}, "csgb_vs_feature_matched_constant_tps_percent": {"mean": 0.2970218703145816, "ci95": [-0.5408266404839033, 1.329329706740591], "documents": 8}, "scope": "Internal training-only4K feasibility continuation; NOT full-study success"}. Requires positive mean and95%CI lower bound for CSGB acceptance and TPS against fit-selected bestfixed, plus TPS against feature-matched constant. This never establishes generalization, an8K gain or the full strict study gate.



Whole-request time includes prefill, fresh attention/features where applicable, Draft/Target work, terminal round and closing synchronization. Canonical AR generation, JSON/audits and external request-end SGD are outside that timer. No explicit per-round synchronization/SGD is used, but item/tolist/D2H and host decisions still cause implicit waits.

Target uses eager attention in FP32; Draft is FP16. Hardware is a server-reported48GB RTX4090. These are instrumented implementation measurements, NOT production-speed evidence. Native fixed paths retain their native attention/cache cadence; attention methods pay their actual fresh-feature cost.

The CPU auditor checks saved chronology, sampling and accounting, not a reconstruction of full GPU attention matrices. Runtime engine/model/input/source manifests bind that limitation explicitly.



## Saved artifacts



Raw results: [/storage/dzj/spec_repro/CSGB_episode_pilot_20261016_v3/pilot](/storage/dzj/spec_repro/CSGB_episode_pilot_20261016_v3/pilot)

[Summary](summary.json) · [Input and artifact hashes](manifest.json) · [Full audit source](analysis_source.py) · [Standalone rerender source](figure_source.py)



[acceptance rendered PNG](acceptance.png) · [vector SVG](acceptance.svg)

[request_tps rendered PNG](request_tps.png) · [vector SVG](request_tps.svg)

[observed_budget_map rendered PNG](observed_budget_map.png) · [vector SVG](observed_budget_map.svg)

[decision_latency rendered PNG](decision_latency.png) · [vector SVG](decision_latency.svg)

[training_learning_curve rendered PNG](training_learning_curve.png) · [vector SVG](training_learning_curve.svg)

[action_stream_variability rendered PNG](action_stream_variability.png) · [vector SVG](action_stream_variability.svg)
