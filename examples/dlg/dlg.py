"""
A federated learning training session with the honest-but-curious server.
The server can analyze periodic gradients from certain clients to
perform the gradient leakage attacks and reconstruct the training data of the victim clients.

Reference:

Zhu et al., "Deep Leakage from Gradients,"
in Advances in Neural Information Processing Systems 2019.

https://papers.nips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf
"""
import dlg_server


def main():
    """ A Plato federated learning training session with the honest-but-curious server. """
    server = dlg_server.Server()
    server.run()


if __name__ == "__main__":
    main()