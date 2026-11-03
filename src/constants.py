"""Constants used in the repository."""

# Fraction of the top activations of a unit that form its activation mask
NETDISSECT_QUANTILE = 0.005

# In vision, the first beam contains FIRST_BEAM_FACTOR * beam_limit concepts
FIRST_BEAM_FACTOR = 2

# Number of closest words in the word2vec space used to build the neighbor
# concepts
NLI_NEIGHBORHOOD_SIZE = 5

# Seed of the data loaders. It is independent of --seed, so the samples are
# always visited in the same order.
DATA_LOADER_SEED = 0

# Number of samples processed together with fast_impl
FAST_IMPL_BATCH_SIZE = 256

# Number of samples whose MMESH rectangles are computed together
MMESH_BATCH_SIZE = 1024

# Number of SNLI sentences per batch. Each sample has two sentences. The BiLSTM
# activations depend slightly on the batch composition.
NLI_BATCH_SIZE = 32

# Number of images shown for each explanation in the vision figures
VISUALIZED_SAMPLES = 4
# Number of premise-hypothesis pairs shown for each explanation in the NLI files
VISUALIZED_SENTENCE_PAIRS = 5
