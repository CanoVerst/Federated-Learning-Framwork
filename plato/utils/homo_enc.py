"""
Utility functions for homomorphic encryption.
"""
import os
import tenseal as ts

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

def remove_context():
    context_dir = ".ckks_context/"
    context_name = "context"
    try:
        os.remove(os.path.join(context_dir, context_name))
        os.remove(context_dir)
    finally:
        pass
