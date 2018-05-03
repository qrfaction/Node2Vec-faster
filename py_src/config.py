import multiprocessing as mlp




class config:

    dimensions = 128

    walk_length = 80

    epoch = 10

    window_size = 10

    workers = mlp.cpu_count()

    batchsize = 100

    p = 1.0

    q = 1.0

    weighted = False

    output_path = "../output/"

