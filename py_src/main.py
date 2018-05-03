import tools
from multiprocessing import Process, Queue
from gensim.models import Word2Vec
from config import config


def train(msg_queue):

    N = tools.get_num_node()
    batchsize = config.batchsize
    epoch = config.epoch
    output_path = config.output_path

    model = Word2Vec(size=config.dimensions, window=config.window_size, min_count=0, sg=1, workers=config.workers)

    model.build_vocab([[str(i) for i in range(N)]])

    iters = epoch*((N + batchsize - 1)//(batchsize))

    for i in range(iters):
        walks = msg_queue.get()
        model.train(walks,epochs=1,total_examples=len(walks))

    model.wv.save_word2vec_format(output_path+'node2vec.emb')

def sampling(msg_queue):
    from sklearn.cross_validation import KFold
    import numpy as np

    N = tools.get_num_node()
    batchsize = config.batchsize
    epoch = config.epoch
    n_folds = (N + batchsize - 1)//(batchsize)
    length = config.walk_length


    kf = KFold(N, n_folds=n_folds, shuffle=True)

    nodes = np.arange(N)

    for i in range(epoch):
        for _,node_idx in kf:

            walks = tools.get_samples_batch(len(node_idx),
                                            length,
                                            nodes[node_idx])
            walks = [list(map(str, walk)) for walk in walks]
            msg_queue.put(walks)

def main():

    q = config.q
    p = config.p
    weighted = config.weighted
    msg_q = Queue()

    with tools.openGraph(q,p,weighted):
        p_train = Process(target=train, args=(msg_q,))
        p_train.start()

        sampling(msg_q)

        p_train.join()


if __name__ == '__main__':
    main()