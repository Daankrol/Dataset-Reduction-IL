# Learning setting
config = dict(
    setting="SL",
    measure_energy=False,
    wandb=True,
    no_tsne=True,
    is_reg=False,
    dataset=dict(
        name="cifar10", datadir="../data", feature="dss", type="image", img_size=32
    ),
    dataloader=dict(shuffle=True, batch_size=32, pin_memory=True),
    model=dict(architecture="ResNet18", type="pre-defined"),
    ckpt=dict(is_load=False, is_save=False, dir="results/", save_every=20),
    loss=dict(type="CrossEntropyLoss", use_sigmoid=False),
    optimizer=dict(type="sgd", momentum=0.9, lr=0.005, weight_decay=5e-4, nesterov=False),
    scheduler=dict(type="cosine_annealing", T_max=300),
    # early_stopping=True,
    dss_args=dict(
        type="CAL",
        fraction=0.1,
        select_every=1,
        selection_type='PerBatch',
        metric='euclidean',
        kappa=0
    ),
    # dss_args=dict(
    #     type="Submodular",
    #     fraction=0.01,
    #     select_every=1,
    #     selection_type='Supervised',  # Can be: 'PerClass', 'Supervised'
    #     submod_func_type='facility-location',
    #     # Can be: 'facility-location', , 'graph-cut', 'sum-redundancy', 'saturated-coverage'
    #     optimizer='two-stage',  # two-stage, random, modular, naive, lazy, greedi etc...
    #     if_convex=False,
    #     linear_layer=False,
    #     kappa=0,
    # ),
    # dss_args=dict(
    #     type="FacLoc",
    #     selection_type='Supervised',
    #     submod_func_type='facility-location',
    #     fraction=0.3,
    #     select_every=1,
    #     kappa=0,
    #     ),
    # dss_args=dict(
    #     type="GradMatch",
    #     fraction=0.5,
    #     select_every=1,
    #     lam=0.5,
    #     selection_type="PerClassPerGradient",
    #     v1=True,
    #     valid=False,
    #     kappa=0,
    #     eps=1e-100,
    #     linear_layer=True,
    # ),
    train_args=dict(
        num_epochs=3,
        device="cuda",
        print_every=1,
        results_dir="../results/",
        print_args=[
            "val_loss",
            "val_acc",
            "val_recall",
            "tst_loss",
            "tst_acc",
            "tst_recall",
            "trn_loss",
            "trn_acc",
            "trn_recall",
            "time",
        ],
        return_args=[],
    ),
)
