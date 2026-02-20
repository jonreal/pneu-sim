## Training Configurations

The following models are trained with different damping values (`B`) and dataset combinations.

---

### Single-Dataset Training (nn_1010)

- **Model 1** — B = 6500, trained on `nn_1010`
- **Model 2** — B = 5000, trained on `nn_1010`
- **Model 3** — B = 4000, trained on `nn_1010`
- **Model 4** — B = 3000, trained on `nn_1010`
- **Model 5** — B = 2000, trained on `nn_1010`
- **Model 6** — B = 1000, trained on `nn_1010`
- **Model 7** — B = 0,    trained on `nn_1010`

---

### Multi-Dataset Training (B = 0)

- **Model 8** — `nn_1010`, `nn_4545`, `nn_8080`  
  *(symmetric stiffness)*

- **Model 9** — `nn_1010`, `nn_4545`, `nn_8080`, `nn_4050`, `nn_5040`

- **Model 10** — `nn_1010`, `nn_4545`, `nn_8080`, `nn_4050`, `nn_5040`,  
  `nn_3060`, `nn_6030`

- **Model 11** — `nn_1010`, `nn_4545`, `nn_8080`, `nn_4050`, `nn_5040`,  
  `nn_3060`, `nn_6030`, `nn_2070`, `nn_7020`

- **Model 12** — `nn_1010`, `nn_4545`, `nn_8080`, `nn_4050`, `nn_5040`,  
  `nn_3060`, `nn_6030`, `nn_2070`, `nn_7020`, `nn_1080`, `nn_8010`

- **Model 13** — Model 12 + `nn_4530`, `nn_4560`, `nn_3045`, `nn_6045`

- **Model 14** — Model 13 + `nn_4520`, `nn_4570`, `nn_2045`, `nn_7045`

- **Model 15** — Model 14 + `nn_4510`, `nn_4580`, `nn_1045`, `nn_8045`

- **Model 16** — Model 15 + `nn_2020`, `nn_3030`, `nn_4040`,  
  `nn_5050`, `nn_6060`, `nn_7070`