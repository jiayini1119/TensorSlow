import sys
sys.path.append('../tensorslow_serving')


import tensorslow_serving as tss

serving = tss.serving.TensorSlowServer(
    host='127.0.0.1:5000', root_dir='./epoches', model_file_name='my_model.json', weights_file_name='my_weights.npz')

serving.serve()