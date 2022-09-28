# Learning setting
config = dict(
    setting="SL",
    measure_energy=True,
    wandb=True,
    is_reg=False,
    dataset=dict(
        name="cub200", datadir="../data", feature="dss", type="image", img_size=224
    ),
    dataloader=dict(shuffle=True, batch_size=32, pin_memory=True),
    model=dict(architecture="EfficientNet", type="pre-defined"),
    ckpt=dict(is_load=False, is_save=False, dir="results/", save_every=20),
    loss=dict(type="CrossEntropyLoss", use_sigmoid=False),
    optimizer=dict(type="sgd", momentum=0.9, lr=0.005, weight_decay=5e-4, nesterov=False),
    scheduler=dict(type="cosine_annealing", T_max=300),
    dss_args=dict(
        type="Super-CL",
        fraction=0.2,
        selection_type="PerClass",  # PerClass or PerBatch
        weighted=False,
        online=True,
        select_every=10,
        kappa=0
    ),
    train_args=dict(
        num_epochs=300,
        device="cuda",
        print_every=1,
        results_dir="../results/",
        print_args=[
            "val_loss",
            "val_acc",
            "tst_loss",
            "tst_acc",
            "trn_loss",
            "trn_acc",
            "time",
            "trn_recall",
            "tst_recall",
            "val_recall",
        ],
        return_args=[],
    ),
)
