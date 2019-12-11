from imageai.Prediction.Custom import ModelTraining

trainer = ModelTraining()
trainer.setModelTypeAsResNet()
trainer.setDataDirectory("data")
trainer.trainModel(num_objects=15, num_experiments=50, enhance_data=True, save_full_model=True, batch_size=25, show_network_summary=True, transfer_from_model="resnet50_weights_tf_dim_ordering_tf_kernels.h5", initial_num_objects=1000, transfer_with_full_training=True)
