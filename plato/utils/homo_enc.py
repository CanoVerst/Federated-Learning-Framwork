"""
Utility functions for homomorphic encryption.
"""
import os
import torch
import tenseal as ts

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

def encrypt_weights(plain_weights, serialize = True, context = None):
    if context == None:
        context = get_ckks_context()

    encrypted_weights = OrderedDict()

    for name, weights in plain_weights.items():
        encrypted_tensor = torch.flatten(weights)
        encrypted_tensor = ts.ckks_vector(context, encrypted_tensor)
        if serialize:
            encrypted_tensor = encrypted_tensor.serialize()
        
        encrypted_weights[name] = encrypted_tensor

    return encrypted_weights

def deserialize_weights(serialized_weights, context):
    if context == None:
        context = get_ckks_context()

    deserialized_weights = OrderedDict()
    for name, weight in serialized_weights.items():
        deser_weight = ts.lazy_ckks_vector_from(weight)
        deser_weight.link_context(context)
        deserialized_weights[name] = deser_weight
    return deserialized_weights

def decrypt_weights(encrypted_weights, weight_shapes = None):
    decrypted_weights = OrderedDict()
    for name, weight in encrypted_weights.items():
        assert name in weight_shapes
        rebuilt_tensor = torch.tensor(weight.decrypt())
        decrypted_weights[name] = rebuilt_tensor.reshape(weight_shapes[name])
    return decrypted_weights