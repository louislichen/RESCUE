"""Pure toy policy/gradient tests: no dataset, model, GPU or efficacy claim."""
import copy
from dataclasses import FrozenInstanceError, replace
import json
import math
import random
import unittest

from csgb_episode_policy import BUDGETS, Decision, EpisodeGradient, EpisodePolicy, episode_gradient


class CountingRandom(random.Random):
    def __init__(self, seed):
        super().__init__(seed)
        self.calls = 0

    def random(self):
        self.calls += 1
        return super().random()


class ZeroRandom(random.Random):
    def random(self):
        return 0.


def csgb(temperature=1.):
    return EpisodePolicy("csgb", ((.1,.02,-.03,.04), (-.1,-.01,.02,.01),
                                  (.03,.01,.01,-.02), (.02,-.02,.03,.01)),
                         temperature, (.2,.1,.02), (.1,.05,.01))


def grad(policy, decisions, **kwargs):
    values = {"useful_n": 8, "elapsed_s": 1., "rho": 2., "baseline": 1.}
    values.update(kwargs)
    return episode_gradient(policy, decisions, **values)


class EpisodePolicyTests(unittest.TestCase):
    def test_stable_softmax_large_common_logits_and_same_api(self):
        policy = EpisodePolicy("constant", ((1000.,),(1001.,),(999.,),(998.,)), 1.)
        probabilities = policy.probabilities([1.,.4,.3,.12])
        expected = [math.exp(v) for v in (-1.,0.,-2.,-3.)]
        expected = [v/sum(expected) for v in expected]
        for a,b in zip(probabilities,expected): self.assertAlmostEqual(a,b,places=15)
        self.assertAlmostEqual(sum(probabilities),1.)
        with self.assertRaises(TypeError): policy.sample([1.,.4,.3,.12],random.Random(1),explore=False)

    def test_request_seed_roundtrip_replay_one_uniform_and_immutable_parameters(self):
        policy = csgb(); initial = copy.deepcopy(policy.to_state()); fingerprint = policy.parameter_sha256
        rng, replay_rng = CountingRandom(123), CountingRandom(123)
        rows = [policy.sample([1.,.4,.3,.12],rng) for _ in range(10)]
        restored = EpisodePolicy.from_state(json.loads(json.dumps(policy.to_state(),allow_nan=False)))
        replayed = [restored.sample([1.,.4,.3,.12],replay_rng) for _ in range(10)]
        self.assertEqual(rows,replayed); self.assertEqual(rng.calls,10)
        self.assertEqual(rng.getstate(),replay_rng.getstate())
        self.assertEqual(policy.to_state(),initial); self.assertEqual(policy.parameter_sha256,fingerprint)
        for row in rows:
            state = json.loads(json.dumps(row.to_state(),allow_nan=False))
            self.assertEqual(policy.replay(Decision.from_state(state)),row)
        self.assertNotIn("rng",initial)
        with self.assertRaises(FrozenInstanceError): policy.temperature = 2.
        with self.assertRaises(TypeError): policy.weights[0][0] = 4.

    def test_startup_has_no_rng_draw_or_score_and_invalid_input_is_not_consumed(self):
        policy, rng = csgb(), CountingRandom(7)
        before = rng.getstate(); row = policy.sample(None,rng)
        self.assertEqual(row.budget,8); self.assertIsNone(row.u); self.assertIsNone(row.transformed_x)
        self.assertEqual(row.probabilities,(1.,0.,0.,0.)); self.assertEqual(rng.calls,0)
        self.assertEqual(rng.getstate(),before)
        self.assertEqual(grad(policy,[row]).values,((0.,)*4,)*4)
        for raw in ([1.,.4,.3,.11], [1.,True,.3,.3], [1.,float("nan"),.3,.1], [1.,.4,.3]):
            with self.subTest(raw=raw), self.assertRaises(ValueError): policy.sample(raw,rng)
        self.assertEqual(rng.calls,0); self.assertEqual(rng.getstate(),before)
        with self.assertRaises(ValueError): policy.sample(None,None)

    def test_tau_scale_raw_interaction_constant_and_history_projection(self):
        raw = [1.,.4,.3,.12]
        policy = csgb(2.)
        transformed = policy.transform(raw)
        for a,b in zip(transformed,(1.,2.,4.,10.)): self.assertAlmostEqual(a,b)
        self.assertNotAlmostEqual(transformed[3],transformed[1]*transformed[2])
        constant = EpisodePolicy("constant",((0.,),(1.,),(2.,),(3.,)),2.)
        self.assertEqual(constant.transform(raw),(1.,))
        self.assertEqual(constant.probabilities(raw),constant.probabilities([1.,.1,.8,.08]))
        history = EpisodePolicy("history_native",((0.,0.,0.),)*4,1.)
        self.assertEqual(history.transform([1.,.2,.4]),(1.,.2,.4))
        with self.assertRaises(ValueError): history.transform(raw)
        with self.assertRaises(ValueError): constant.transform([1.,.4,.3])
        self.assertNotEqual(csgb(1.).probabilities(raw),policy.probabilities(raw))

    def test_explicit_sgd_new_version_and_stale_gradient_rejection(self):
        policy = csgb(); before = policy.to_state(); fingerprint = policy.parameter_sha256
        gradient = grad(policy,[policy.sample([1.,.4,.3,.12],random.Random(4))])
        updated = policy.sgd(gradient,.01)
        self.assertEqual(policy.to_state(),before); self.assertEqual(policy.parameter_sha256,fingerprint)
        self.assertEqual(updated.parameter_version,policy.parameter_version+1)
        self.assertNotEqual(updated.parameter_sha256,fingerprint)
        for row,w,g in zip(updated.weights,policy.weights,gradient.values):
            for a,b,c in zip(row,w,g): self.assertAlmostEqual(a,b+.01*c)
        with self.assertRaises(ValueError): updated.sgd(gradient,.01)
        with self.assertRaises(ValueError): policy.sgd(replace(gradient,parameter_version=9),.01)
        with self.assertRaises(ValueError): policy.sgd(EpisodeGradient(fingerprint,0,((1.,),)*4),.01)
        for eta in (0.,-.1,float("nan"),True):
            with self.subTest(eta=eta), self.assertRaises(ValueError): policy.sgd(gradient,eta)

    def test_gradient_likelihood_finite_differences_with_nonunit_temperature(self):
        policy = csgb(1.7)
        rows = [policy.sample(raw,random.Random(seed)) for raw,seed in
                (([1.,.4,.3,.12],3),([1.,.5,.2,.1],9))]
        gradient = grad(policy,rows).values
        def objective(subject):
            return 5.*sum(math.log(subject.probabilities(r.raw_x)[BUDGETS.index(r.budget)]) for r in rows)
        step = 1e-6
        for k in range(4):
            for j in range(4):
                plus,minus = [list(r) for r in policy.weights],[list(r) for r in policy.weights]
                plus[k][j] += step; minus[k][j] -= step
                difference = (objective(replace(policy,weights=plus))-objective(replace(policy,weights=minus)))/(2*step)
                self.assertAlmostEqual(gradient[k][j],difference,delta=2e-7)

    def test_repeated_decision_doubles_gradient_not_round_mean(self):
        policy = EpisodePolicy("constant",((0.,),)*4,1.)
        row = policy.sample([1.,.4,.3,.12],ZeroRandom())
        one,two = grad(policy,[row]).values,grad(policy,[row,row]).values
        self.assertEqual(one,((3.75,),(-1.25,),(-1.25,),(-1.25,)))
        self.assertEqual(two,tuple(tuple(2*x for x in r) for r in one))
        startup = policy.sample(None,random.Random(0))
        self.assertEqual(grad(policy,[startup,row]).values,one)
        self.assertEqual(grad(policy,[]).values,((0.,),)*4)

    def test_variable_horizon_expected_return_gradient_not_round_normalized(self):
        """SYNTHETIC UNIT TEST: enumerate all13 paths, never a model experiment."""
        policy = csgb(1.7)
        first_x, second_x = (1.,.4,.3,.12), (1.,.5,.2,.1)
        second_costs = (.3,.7,1.5,2.)

        def decision(subject, raw, arm):
            probabilities = subject.probabilities(raw)
            u = math.fsum(probabilities[:arm])+probabilities[arm]/2.
            row = Decision(subject.parameter_sha256,subject.parameter_version,raw,
                           subject.transform(raw),probabilities,u,BUDGETS[arm])
            return subject.replay(row)

        def trajectories(subject):
            first_p, second_p = subject.probabilities(first_x), subject.probabilities(second_x)
            # First8 stops at H1 and commits2; all other first actions commit1
            # then a second action commits1. N=2 for every trajectory.
            paths = [(first_p[0],1.,[decision(subject,first_x,0)])]
            for first_arm in range(1,4):
                for second_arm in range(4):
                    paths.append((first_p[first_arm]*second_p[second_arm],
                                  .2+second_costs[second_arm],
                                  [decision(subject,first_x,first_arm),
                                   decision(subject,second_x,second_arm)]))
            return paths

        paths = trajectories(policy)
        self.assertEqual(len(paths),13)
        self.assertEqual(sorted(len(rows) for _,_,rows in paths),[1]+[2]*12)
        self.assertAlmostEqual(math.fsum(prob for prob,_,_ in paths),1.)
        weighted = [[0.]*4 for _ in range(4)]
        wrong_round_mean = [[0.]*4 for _ in range(4)]
        for probability,seconds,rows in paths:
            gradient = episode_gradient(policy,rows,useful_n=2,elapsed_s=seconds,
                                        rho=1.,baseline=0.).values
            for k in range(4):
                for j in range(4):
                    weighted[k][j] += probability*gradient[k][j]
                    wrong_round_mean[k][j] += probability*gradient[k][j]/len(rows)

        def expected_return(subject):
            return math.fsum(probability*(2.-seconds)
                             for probability,seconds,_ in trajectories(subject))

        step = 1e-6
        for k in range(4):
            for j in range(4):
                plus,minus = [list(r) for r in policy.weights],[list(r) for r in policy.weights]
                plus[k][j] += step; minus[k][j] -= step
                finite_difference = (expected_return(replace(policy,weights=plus))
                                     -expected_return(replace(policy,weights=minus)))/(2*step)
                self.assertAlmostEqual(weighted[k][j],finite_difference,delta=2e-8)
        self.assertGreater(max(abs(weighted[k][j]-wrong_round_mean[k][j])
                               for k in range(4) for j in range(4)),1e-3)

    def test_decision_strict_u_probability_action_shape_and_parameter_binding(self):
        policy = csgb(); row = policy.sample([1.,.4,.3,.12],ZeroRandom())
        for updates in ({"u":1.},{"u":-.1},{"u":float("nan")},{"u":True},
                        {"budget":16},{"budget":True},{"probabilities":(.2,.2,.2,.2)},
                        {"probabilities":(1.,0.,0.,0.)},{"raw_x":(1.,.4,.3,.11)},
                        {"transformed_x":(1.,2.)}):
            with self.subTest(updates=updates), self.assertRaises(ValueError): replace(row,**updates)
        for updates in ({"parameter_version":1},{"parameter_sha256":"0"*64},
                        {"transformed_x":(1.,0.,0.,0.)}, {"probabilities":(.4,.3,.2,.1)}):
            with self.subTest(updates=updates), self.assertRaises(ValueError): policy.replay(replace(row,**updates))
        state = row.to_state(); state["source_schema"] = "unknown"
        with self.assertRaises(ValueError): Decision.from_state(state)
        with self.assertRaises(ValueError): policy.replay(row.to_state())

    def test_serialization_deep_copy_and_rejects_unknown_schema_rng_or_native_kv(self):
        policy = csgb(); fingerprint = policy.parameter_sha256
        state = policy.to_state(); state["weights"][0][0] += 3
        self.assertEqual(policy.parameter_sha256,fingerprint)
        for key,value in (("schema",True),("schema",2),("source_schema","unknown"),("native_v",False),
                          ("fallback",True),("budgets",[8.,16,24,32]),("parameter_version",.1),
                          ("rng",{}),("temperature",0.),("weights",[[0.]*3]*4)):
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                EpisodePolicy.from_state({**policy.to_state(),key:value})
        with self.assertRaises(ValueError): EpisodePolicy("csgb",((0.,)*4,)*4,1.,(0.,)*3,(1.,0.,1.))
        with self.assertRaises(ValueError): EpisodePolicy("constant",((0.,),)*4,1.,(0.,)*3,(1.,)*3)
        with self.assertRaises(ValueError): EpisodePolicy("history_native",((float("nan"),0.,0.),)*4,1.)
        with self.assertRaises(ValueError): EpisodePolicy("constant",((1000.,),(-1000.,),(0.,),(0.,)),1.).probabilities([1.,.4,.3,.12])

    def test_invalid_episode_counts_baseline_time_and_foreign_parameters_rejected(self):
        policy = csgb(); row = policy.sample([1.,.4,.3,.12],random.Random(2))
        for name,value in (("useful_n",0),("useful_n",8.),("useful_n",True),("useful_n",2**10000),
                           ("elapsed_s",0.),("elapsed_s",float("nan")),("rho",-1.),("baseline",float("inf"))):
            with self.subTest(name=name), self.assertRaises(ValueError): grad(policy,[row],**{name:value})
        with self.assertRaises(ValueError): grad(policy,[row.to_state()])
        with self.assertRaises(ValueError): grad(replace(policy,parameter_version=1),[row])
        with self.assertRaises(ValueError): grad(policy,[replace(row,parameter_sha256="0"*64)])


if __name__ == "__main__": unittest.main()
