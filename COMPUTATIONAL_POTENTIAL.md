# Computational Potential

This repository now has a shared cost currency for the request-level
`dynamic_resource` router. It preserves the legacy width proxy by default,
but can use measured hardware costs:

```text
E(x, G) = task_loss(x, G) + lambda * C(G)
C(G)   = sum(action_cost) + fixed_cost + group_cost
```

The action profile is ordered like `[0, 64, 128, 256, 512]`. It does not have
to be linear in width. The calibration tool measures each homogeneous route
on the target GPU and subtracts the zero-width route, producing incremental
milliseconds per layer:

```powershell
python experiments/calibrate_computational_potential.py `
  --checkpoint=out-shakespeare-char-dynamic-resource/ckpt.pt
```

Use the generated JSON while evaluating one checkpoint at several operating
points:

```powershell
python experiments/computational_potential_sweep.py `
  --checkpoint=out-shakespeare-char-dynamic-resource/ckpt.pt `
  --cost-profile=out-shakespeare-char-dynamic-resource/computational_potential_profile.json `
  --lambdas=0,0.1,0.25,0.5,1,2
```

For training, set these config values:

```python
computational_potential_enabled = True
computational_potential_lambda = 0.05
computational_potential_cost_profile = [0.0, 0.12, 0.19, 0.34, 0.61]
computational_potential_fixed_cost = 0.0
computational_potential_group_cost = 0.02
```

The same checkpoint can be priced at inference with `--potential-lambda`, so
the lambda sweep does not require retraining. The current implementation
fully wires this into `dynamic_resource`; the other MLP variants still expose
their legacy individual loss weights and should be migrated only after their
action spaces have a measured cost profile.
