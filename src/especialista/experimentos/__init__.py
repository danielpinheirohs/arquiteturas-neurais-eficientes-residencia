"""Os experimentos de cada etapa: quais variantes rodar, com cache em disco.

Treinar custa minutos; ler um resultado custa milissegundos. Cada execução
grava o que produziu em `data/execucoes/` e só roda de novo se a pasta for
apagada. Os notebooks pedem a execução e recebem o resultado, sem saber se
ela acabou de rodar ou já estava pronta.
"""
