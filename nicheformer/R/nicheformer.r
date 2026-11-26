## Nicheformer embeddings in R
##
## Runs the TorchScript model exported from nicheformer/Python/nicheformer.ipynb
## ("TorchScript export" section) on the pre-tokenized cells. No Python needed.
##
## Inputs (both written by the notebook):
##   nicheformer_embedder.pt  compiled model
##   tokens.pt                tokenized cells: X [n_cells, 1500] + modality, specie, assay
## Output:
##   emb                      matrix [n_cells, 512], one row per cell
##
## Paths are relative to the repo root; run the script from there.

library(torch)

# Computes embeddings in batches.
#   model       TorchScript module from jit_load()
#   tokens      named list of tensors from tokens.pt; each is sliced along dim 1 (cells)
#   batch_size  cells per forward pass
#   device      "cpu" or "cuda"
# Returns a matrix [n_cells, 512].
run_nicheformer <- function(model, tokens, batch_size = 8L, device = "cpu") {
  n <- tokens$X$size(1)
  out <- list()
  model$to(device = device)
  with_no_grad({
    for (start in seq(1L, n, by = batch_size)) {
      end <- min(start + batch_size - 1L, n)
      batch <- lapply(tokens, function(t) t[start:end]$to(device = device))
      out[[length(out) + 1]] <- as.matrix(model(batch)$cpu())
      cat(sprintf("\r%d / %d cells", end, n))
    }
  })
  cat("\n")
  do.call(rbind, out)
}

# Load the tokens (a dict of tensors saved with torch.save)
tokens <- load_state_dict("./nicheformer/Python/tokens.pt")
names(tokens)        # "X" "modality" "specie" "assay"
tokens$X$shape       # 100 1500

# Load the compiled model
model <- jit_load("./nicheformer/Python/nicheformer_embedder.pt")

# Compute embeddings (use device = "cpu" if no GPU is available)
t0 <- Sys.time()
emb <- run_nicheformer(model, tokens, batch_size = 4L, device = "cuda")
model_time <- Sys.time() - t0

model_time
mean(emb)            # should match the mean in the notebook (0.0183)
