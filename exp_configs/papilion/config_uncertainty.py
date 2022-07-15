# Learning setting
config = dict(
    setting="SL",
    measure_energy=True,
    wandb=True,
    is_reg=False,
    dataset=dict(name="papilion", datadir="../data", feature="dss", type="image", img_size=224),
    dataloader=dict(shuffle=True, batch_size=20, pin_memory=True),
    model=dict(architecture="EfficientNet", type="pre-trained", fine_tune=True),
    ckpt=dict(is_load=False, is_save=False, dir="results/", save_every=20),
    loss=dict(type="CrossEntropyLoss", use_sigmoid=False),
    optimizer=dict(
        type="adam", lr=0.005
    ),
    scheduler=dict(type="cosine_annealing", T_max=300),
    dss_args=dict(
        type="Uncertainty",
        fraction=0.1,
        select_every=10,
        selection_type='LeastConfidence',
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
