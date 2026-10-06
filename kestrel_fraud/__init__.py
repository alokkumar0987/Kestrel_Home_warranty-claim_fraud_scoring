"""Kestrel Home warranty-claim fraud scoring.

Modules, in pipeline order:
    config      paths, business rules, modelling choices
    data        load the data pack, clean claims
    features    point-in-time features
    model       estimator, training weights, score correction, saved bundle
    explain     plain-language reasons
    validation  time-based split and review-desk metrics
    train       CLI: fit on all labelled claims, write predictions
    evaluate    CLI: write the validation report
    service     FastAPI app and screen
"""
__version__ = "1.1.0"
