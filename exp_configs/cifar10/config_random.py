# Learning setting
config = dict(
    setting="SL",
    is_reg=False,
    wandb=True,
    measure_energy=True,
    dataset=dict(
        name="cifar10", datadir="../data", feature="dss", type="image", img_size=224
    ),
    dataloader=dict(shuffle=True, batch_size=32, pin_memory=True),
    model=dict(architecture="EfficientNet", type="pre-defined"),
    ckpt=dict(is_load=False, is_save=False, dir="results/", save_every=20),
    loss=dict(type="CrossEntropyLoss", use_sigmoid=False),
    optimizer=dict(type="adam",lr=0.01),
    scheduler=dict(type="cosine_annealing", T_max=300),
    dss_args=dict(type="Random", fraction=0.1, select_every=10, kappa=0, online=True),
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
