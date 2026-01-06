# Autoencoder + NN (ML-CUP)

Script standalone per autoencoder + regressore con grid search su latent size, dropout e L2.

## File principale

- `AE_NN.py`: carica il dataset, esegue lo split 60/20/20, fa grid search e retraining finale.

## Esecuzione

Modifica queste variabili in `AE_NN.py` prima del run:

- `DATA_PATH`: path del dataset TR.
- `DEVICE`: `"cpu"` o `"cuda"`.

Run:

```bash
python Cup\Leo\MLP\Autoencoder+NN\AE_NN.py
```

## Output

Il run genera una cartella in `results/` con:

- `grid_search_results.csv`
- `latent_dim_sweep.png`
- `final_model_learning_curve.png`
- `final_model_hyperparams.json`
- `run_summary.json`
- `ae_nn_best.pt`

Inoltre salva il modello migliore (per l'ensamble) in:

- `Cup/Leo/Ensemble/models/ae_nn/ae_nn_best.pt`
