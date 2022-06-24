"""
Utility functions for homomorphic encryption.
"""
import os
import torch
import tenseal as ts
import numpy as np
import random

from typing import OrderedDict

def get_ckks_context():
    context_dir = ".ckks_context/"
    context_name = "context"
    try:
        with open(os.path.join(context_dir, context_name), "rb") as f:
            return ts.context_from(f.read())
    except:
        if not os.path.exists(context_dir):
            os.mkdir(context_dir)
        
        context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
        # context.generate_galois_keys()
        context.global_scale = 2**40

        with open(os.path.join(context_dir, context_name), "wb") as f:
            f.write(context.serialize(save_secret_key=True))
            f.close()

        return context

def encrypt_weights(plain_weights, serialize = True, context = None, para_nums = None, encrypt_ratio = 0.05):
    if context == None:
        context = get_ckks_context()
    
    output = OrderedDict()
    encrypt_indices = []
    total_para_num = 0
    sample_size = 0

    # Create a list of indices with length of sample_size which is computed based on
    # the number of parameters of the model and the encrypt_ratio
    for value in para_nums.values():
        total_para_num += value
    sample_size = int(total_para_num*encrypt_ratio)
    encrypt_indices = random.sample(range(0,total_para_num), sample_size)

    # Flatten all weight tensors to a vector
    flattened_weight_arr = np.array([])
    for weight in plain_weights.values():
        flattened_weight_arr = np.append(flattened_weight_arr, weight)
    weights_vector = torch.from_numpy(flattened_weight_arr)

    if encrypt_ratio == 0:
        output['unencrypted_weights'] = weights_vector
        output['encrypt_indices'] = None
        output['encrypted_weights'] = None
        return output

    # Create a vector of weights selected by the encrypt_indices
    weights_to_enc = weights_vector[encrypt_indices]

    # Set weights selected by encrypt_indices of the model to be 0
    weights_vector[encrypt_indices] = 0

    # Encrypt selected weight vector
    unencrypted_weights = weights_vector
    encrypted_weights = ts.ckks_vector(context, weights_to_enc)

    # Serialize the encrypted weight vector if indicated
    if serialize:
        encrypted_weights = encrypted_weights.serialize()

    output['unencrypted_weights'] = unencrypted_weights
    output['encrypted_weights'] = encrypted_weights
    output['encrypt_indices'] = encrypt_indices

    return output

def deserialize_weights(serialized_weights, context):
    if context == None:
        context = get_ckks_context()
    deserialized_weights = OrderedDict()
    for name, weight in serialized_weights.items():
        if name == 'encrypted_weights' and weight != None:
            deser_weights_vector = ts.lazy_ckks_vector_from(weight)
            deser_weights_vector.link_context(context)
            deserialized_weights[name] = deser_weights_vector
        else:
            deserialized_weights[name] = weight

    return deserialized_weights

def decrypt_weights(encrypted_weights, weight_shapes = None, para_nums = None):
    decrypted_weights = OrderedDict()
    vector_length = []
    for para_num in para_nums.values():
        vector_length.append(para_num)
    
    # Decrypt the encrypted sample vector
    encrypt_indices = encrypted_weights['encrypt_indices']
    unencrypted_weights = encrypted_weights['unencrypted_weights']
    if encrypted_weights['encrypted_weights'] != None:
        decrypted_sample_vector = torch.tensor(encrypted_weights['encrypted_weights'].decrypt())
    else:
        decrypted_sample_vector = encrypted_weights['encrypted_weights']

    # Rebuild the original weight vector by returning selected values
    if encrypt_indices != None:
        sample_index = 0
        for index in encrypt_indices:
            unencrypted_weights[index] = decrypted_sample_vector[sample_index]
            sample_index += 1

    # Convert flattened weight vector back to a dict of weight tensors
    decrypted_weights_vector = unencrypted_weights
    decrypted_weights_vector = torch.split(decrypted_weights_vector, vector_length)
    weight_index = 0
    for name, shape in weight_shapes.items():
        decrypted_weights[name] = decrypted_weights_vector[weight_index].reshape(shape)
        weight_index = weight_index + 1

    return decrypted_weights