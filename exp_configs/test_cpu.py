# Learning setting
config = dict(
    setting="SL",
    measure_energy=False,
    wandb=True,
    is_reg=False,
    dataset=dict(
        name="papilion", datadir="../data", feature="dss", type="image", img_size=224
    ),
    dataloader=dict(shuffle=True, batch_size=20, pin_memory=True),
    model=dict(architecture="EfficientNet", type="pre-defined"),
    ckpt=dict(is_load=False, is_save=False, dir="results/", save_every=20),
    loss=dict(type="CrossEntropyLoss", use_sigmoid=False),
    optimizer=dict(
        type="sgd", momentum=0.9, lr=0.01, weight_decay=5e-4, nesterov=False
    ),
    scheduler=dict(type="cosine_annealing", T_max=300),
    # early_stopping=True,
    dss_args=dict(
        type="GradMatch",
        # selection_type="LeastConfidence",
        selection_type="PerClassPerGradient",
        v1=True,
        lam=0.5,
        valid=False,
        linear_layer=True,
        fraction=0.01,
        select_every=1,
        kappa=0,
        eps=1e-100,
        # online=True
    ),
    train_args=dict(
        num_epochs=2,
        device="cpu",
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
