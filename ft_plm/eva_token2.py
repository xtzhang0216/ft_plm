import os
import gin
import esm
from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
from samplers import Sampler
from omegafold.__main__ import main as fold
from omegafold.__main__ import model
from aim import Run as Register
from aim import Text
from pathlib import Path
from metrics.metrics import hamming_distance, perplexity
from utils import gin_config_to_dict, load_fasta_file, save_fasta_file

from samplers import (
    VanillaSampler,
    NucleusSampler,
    GibbsSampler,
    MetropolisHastingsSampler,
    MetropolisSampler,
)
def run(
    fasta_dir: str,
    sampler_class: Sampler,
    n_steps: int = gin.REQUIRED,
    fold_every: int = gin.REQUIRED,
    experiment: str = gin.REQUIRED,
    repo: str = gin.REQUIRED,
):
    repo = str(Path(repo).expanduser())

    # load ESM to memory
    logging.info("loading ESM2")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    # load omegafold to memory
    logging.info("loading Omegafold")
    folder = model()

    # run all experiments in FASTA directory
    fasta_dir = Path(fasta_dir).expanduser()
    fasta_files = [f for f in os.listdir(fasta_dir) if '.fasta' in f]

    # set up Aim run where we keep track of metrics
    register = Register(experiment=experiment, repo=repo)
    register["hparams"] = gin_config_to_dict(gin.config._OPERATIVE_CONFIG)

    # save files in the same path as Aim, using the hash as dir
    register_dir = str(Path(repo) / register.hash)
    os.makedirs(register_dir, exist_ok=False)
    logging.info(f'Saving Structures to {register_dir}')

    for f in fasta_files:
        context = {'fasta': f}
        sampler = sampler_class(esm_model, alphabet)
        logging.info(f"sampling with : {sampler}")
        fasta_file = os.path.join(fasta_dir, f)
        logging.info(f"Running experiment for {fasta_file}")
        sequences, names = load_fasta_file(fasta_file)

        res_sequences = []

        # sample n times from each sampler
        for step in tqdm(range(n_steps)):
            # perform sampler forward
            output_sequences, sample_metrics = sampler.step(sequences)
            res_sequences.append(output_sequences)

            for key, value in sample_metrics.items():
                register.track(value, name=key, step=step, context=context)
            register.track(
                Text(output_sequences[0]), name="sequence", step=step, context=context)

            # calculate hamming distance
            hamming = hamming_distance(res_sequences[0][0], output_sequences[0])
            register.track(hamming, name='hamming_distance', step=step, context=context)

            # calculate perplexity
            ppl = perplexity(output_sequences[0], sampler)