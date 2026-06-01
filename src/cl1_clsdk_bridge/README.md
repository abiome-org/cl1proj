# CL SDK Bridge

`cl1_clsdk_bridge` contains the thin adapters that let local experiment
packages run behind the vendored CL SDK simulator.

The bridge keeps CL SDK integration separate from the SNN reset simulator so
the core experiment code can evolve without becoming mixed into the static SDK
runtime tree.
