import tensorflow as tf

def main():
    with tf.Session() as sess:
        # Restore the graph
        saver = tf.train.import_meta_graph("./model.meta", clear_devices=True)

        # Load weights
        model_path = tf.train.latest_checkpoint("./")
        saver.restore(sess, model_path)

        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            ['y', 'accuracy']
            )

        tf.train.write_graph(frozen_graph_def, './', 'frozen_graph.pb',  as_text=False)
        tf.train.write_graph(frozen_graph_def, './', 'frozen_graph.txt', as_text=True)

if __name__ == '__main__':
    main()