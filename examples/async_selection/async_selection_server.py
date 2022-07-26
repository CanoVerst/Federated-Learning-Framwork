"""
A customized server with asynchronous client selection
"""
import logging

from turtle import up, update
from plato.config import Config
from plato.servers import fedavg
from cvxopt import matrix, log, solvers, sparse
from collections import OrderedDict
from plato.config import Config
import mosek
import numpy as np
import torch


class Server(fedavg.Server):
    """A federated learning server."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)

        self.number_of_client = Config().clients.total_clients
        self.local_gradient_bounds = None
        self.aggregation_weights = None
        self.local_stalenesses = None
        self.squared_deltas_current_round = None

    def configure(self):
        """ Initializes necessary variables. """
        super().configure()

        self.local_gradient_bounds = 0.01 * np.ones(
            self.number_of_client
        )  # 0.01 is a hyperparameter that's used as a starting point.
        self.local_stalenesses = np.zeros(self.number_of_client)
        self.aggregation_weights = np.ones(
            self.number_of_client) * (1.0 / self.number_of_client)

    def choose_clients(self, clients_pool, clients_count):
        """ Choose a subset of the clients to participate in each round. """
        assert clients_count <= len(clients_pool)

        # Select clients based on calculated probability (within clients_pool)

        p = self.calculate_selection_probability(clients_pool)
        print("The calculated probability is: ", p)
        print("current clients pool: ", clients_pool)

        selected_clients = np.random.choice(clients_pool,
                                            clients_count,
                                            replace=False,
                                            p=p)

        logging.info("[%s] Selected clients: %s", self, selected_clients)
        print("type of selected clients: ", type(selected_clients))
        return selected_clients.tolist()

    def compute_weight_deltas(self, updates):
        """Extract the model weight updates from client updates and compute local update bounds with respect to each clients id."""
        weights_deltas = super().compute_weight_deltas(updates)

        #weights_received = [payload for (__, __, payload, __) in updates]
        # used as update list
        self.squared_deltas_current_round = np.zeros(self.number_of_client)
        # the order of id should be same as weights_delts above
        id_received = [client_id for (client_id, __, __, __) in updates]
        # find delat bound for each client.
        for client_id, delta in zip(id_received, weights_deltas):
            #print("what's an orderdict of deltas like: ", delta)
            #self.squared_deltas_current_round[client_id - 1]  #
            # calculate the largest value in each layer and sum them up for the bound
            for layer, value in delta.items():
                if 'conv' in layer: 
                    temp_max = torch.max(value).detach().cpu().numpy()
                    temp_max_abs = np.absolute(temp_max)
                    #print(value)
                    #print("max_abs: ", temp_max_abs)
                    if temp_max_abs > self.squared_deltas_current_round[client_id -
                                                    1]:
                    
                        self.squared_deltas_current_round[client_id -
                                                    1] = temp_max_abs

            #self.squared_deltas_current_round[client_id - 1] = np.square(
            #torch.max(delta).detach().cpu().numpy())
        print("!!!!!!The squared deltas bound of this round are: ",
              self.squared_deltas_current_round)
        return weights_deltas  #, squared_deltas

    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        print("start aggregating weights")
        deltas = await self.federated_averaging(updates)
        updated_weights = self.algorithm.update_weights(deltas)
        self.algorithm.load_weights(updated_weights)
        print("start updating records...")
        self.update_records(updates)

    def update_records(self, updates):
        """"Update clients record on the server"""
        # Extract the local staleness and update the record of client staleness.
        for (client_id, __, __, client_staleness) in updates:
            self.local_stalenesses[client_id - 1] = client_staleness
        print("!!!The staleness of this round are: ", self.local_stalenesses)
        # Update local gradient bounds
        self.local_gradient_bounds = self.squared_deltas_current_round
        print("local_gradient_bounds: ", self.local_gradient_bounds)
        self.extract_aggregation_weights(updates)

    def extract_aggregation_weights(self, updates):
        """Extract aggregation weights"""

        # below is for fedavg only; complex ones would be added later
        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (__, report, __, __) in updates])

        for client_id, report, __, __ in updates:
            self.aggregation_weights[
                client_id - 1] = report.num_samples / self.total_samples
        print("!!!!!!The aggregation weights of this round are: ",
              self.aggregation_weights)

    def calculate_selection_probability(self, clients_pool):
        """Calculte selection probability based on the formulated geometric optimization problem
            Minimize \alpha \sum_{i=1}^N \frac{p_i^2 * G_i^2}{q_i} + A \sum_{i=1}^N q_i * \tau_i * G_i
            Subject to \sum_{i=1}{N} q_i = 1
                    q_i > 0 
            Probability Variables are q_i

        """
        print("Calculating selection probabitliy ... ")
        alpha = 1
        BigA = 1
        # extract info for clients in the pool
        clients_pool = [
            index - 1 for index in clients_pool
        ]  # index in clients_pool starts from 1 rather than 0 in other np.arrays.
        num_of_clients_inpool = len(clients_pool)

        aggregation_weights_inpool = self.aggregation_weights[clients_pool]
        local_gradient_bounds_inpool = self.local_gradient_bounds[clients_pool]
        local_staleness_inpool = self.local_stalenesses[clients_pool]

        # read aggre_weight from somewhere
        aggre_weight_square = np.square(aggregation_weights_inpool)  # p_i^2
        local_gradient_bound_square = np.square(
            local_gradient_bounds_inpool)  # G_i^2

        f1_params = alpha * np.multiply(
            aggre_weight_square, local_gradient_bound_square)  # p_i^2 * G_i^2
        f1 = matrix(np.eye(num_of_clients_inpool) * f1_params)

        f2_params = BigA * np.multiply(
            local_staleness_inpool,
            local_gradient_bounds_inpool)  # \tau_i * G_i
        f2 = matrix(-1 * np.eye(num_of_clients_inpool) * f2_params)

        F = sparse([[f1, f2]])

        g = log(matrix(np.ones(2 * num_of_clients_inpool)))

        K = [2 * num_of_clients_inpool]
        G = matrix(-1 * np.eye(num_of_clients_inpool))  #None
        h = matrix(np.zeros((num_of_clients_inpool, 1)))  #None

        A1 = matrix([[1.]])
        A = matrix([[1.]])
        for i in range(num_of_clients_inpool - 1):
            A = sparse([[A], [A1]])

        b = matrix([1.])
        solvers.options['maxiters']=500
        sol = solvers.gp(
            K, F, g, G, h, A, b,
            solver='mosek')['x']  # solve out the probabitliy of each client

        return np.array(sol).reshape(-1)
