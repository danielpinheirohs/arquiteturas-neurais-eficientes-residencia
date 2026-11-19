"""As LLMs do projeto: incumbentes (o sistema a substituir) e professoras (quem rotula o treino).

As duas LLMs, a fechada e a aberta, recebem o mesmo prompt e passam pela
mesma leitura de resposta:

    demonstracoes.py   os poucos exemplos humanos que a LLM vê, e a escolha dos mais parecidos
    prompt.py          as instruções, o formato da resposta e a mensagem de cada pedido
    resposta.py        do JSON da LLM para uma `Previsao` (onde cada entidade está no pedido)
    claude.py          a LLM fechada: chamada direta, chamada em lote, registro em disco
"""
