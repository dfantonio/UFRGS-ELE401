Prof. Dr.Tiago Oliveira Weber-
Tópicos Especiais em Instrumentação

# Trabalho - Pruning,

# Quantização

### PROF. DR.TIAGOOLIVEIRAWEBER

Programa de Pos ́ -Graduac ̧ ̃ao em Engenharia
Eletrica ́ - PPGEE - Lab. IEE-IA
PortoAlegre- RS - Brasil
Tópicos Especiais em Instrumentação
Inteligência Computacional em Hardware

## 1. Objetivo Geral

Desenvolver, implementar e analisar modelos inteligentes em um sistema micro-
controlado para encontrar o melhor compromisso entre uso de recursos e desempenho
para determinada aplicação.

## 2. Objetivos Específicos

- Analisar e, opcionalmente, desenvolver _dataset_ que utilize dados obtidos através
  de sensores;
- Avaliar variações de modelos inteligentes para o problema considerando com-
  plexidade e utilizando pruning e quantização como recurso;
- Implementar o(s) melhor(es) modelo(s) em sistema microcontrolado e avaliar
  taxa de acerto (ou erro) e latência.

  2.1. Descrição

Neste trabalho, o aluno desenvolverá um sistema microcontrolado com inteligência
computacional para solução de um problema que utilize informações provenientes de
sensores. Para tal, além dos estudos sobre o contexto do problema em si, será realizada
análise a respeito do impacto de estratégias de pruning e quantização. O aluno pode
optar por usar base de dados pronta ou criar a sua (caso tenha acesso aos sensores e
faça um projeto de experimentos).
Assim, este trabalho será também o ponto de partida para uma análise detalhada de
uma base de dados de escolha do aluno (que poderá, a depender da complexidade, ser
utilizada para os próximos trabalhos). O aluno poderá optar por alguma das técnicas
(ou mais de uma) abordadas durante a disciplina (redes neurais, algoritmos baseados
em árvore de decisão ou outras) para ser usada no trabalho.

2.2. Avaliação

O trabalho deverá ser entregue de forma bem estruturada e na forma de artigo para
revista. O artigo poderá ser escrito em português ou inglês e deverá ter:

- Resumo ( _Abstract_ ): descreve o que foi realizado de forma objetiva e sucinta (de
  150 até 250 palavras). Deve incluir: propósito/contexto, métodos, resultados
  quantitativos e conclusão;
- Introdução ( _Introduction_ ): descreve o contexto do problema, discute as aborda-
  gens existentes na literatura e indica o objetivo do trabalho, abordagem e con-
  tribuições;

Prof. Dr.Tiago Oliveira Weber-
Tópicos Especiais em Instrumentação

- Materiais e Métodos ( _Material and methods_ ): detalha como o trabalho foi real-
  izado, quais materiais (incluindo recursos computacionais e softwares) foram uti-
  lizados e os procedimentos realizados para obtenção dos resultados. Em essência,
  permite que o trabalho seja reproduzido por outros pesquisadores;
- Resultados ( _Results_ ): apresentação dos resultados dos experimentos descritos na
  seção "Materiais e Métodos".
- Discussão ( _Discussion_ ): pode estar junto com a seção "Resultados" (formando uma
  seção Resultados e Discussão) caso seja desejado. Esta seção discute e mostra a
  importância dos resultados obtidos.
- Conclusão ( _Conclusion_ ): recapitular o objetivo do trabalho deixando clara a con-
  tribuição do trabalho a partir do que foi obtido, tornar claras limitações, indicar
  futuros trabalhos.

Observe a clara separação entre métodos e resultados. Não adicione resultados na
seção de métodos ou vice-versa. Para templates , consultar o professor ou o material
disponibilizado na disciplina.

## 3. Atividades

Apesar de serem listados diversos itens nesta seção, observe que eles não devem
estar apresentados desta forma no artigo, e sim na estrutura de trabalho mostrada pre-
viamente. Estes itens servem para indicar as atividade esperadas no trabalho entregue.
Os itens podem servir como um guia durante a execução ou um checklist uma vez que
o trabalho esteja pronto. Itens:

- encontre uma base de dados que seja citável (tenha publicação associada a ela) e
  tenha sido utilizada por pelo menos 3 artigos científicos na literatura; Alternati-
  vamente, desenvolva a sua através de um projeto de experimentos.
- faça análise da base de dados (usar ferramentas/pacotes estatísticos para analisá-
  la). Avaliar necessidade de balanceamento de base de dados.
- realize pré-processamento e normalização;
- faça análise de correlação entre os atributos de entrada e avalie a necessidade
  de todas as entradas existirem. Opcionalmente, use estratégias para seleção de
  _features_.
- separe os dados em conjunto treinamento, validação e teste. Opcionalmente,
  use _cross-validation_ para fazer treinamento+validação em um mesmo conjunto
  (preservando o conjunto teste em separado). Mantenha esta mesma separação
  para todos os testes futuros.

Prof. Dr.Tiago Oliveira Weber-
Tópicos Especiais em Instrumentação

- faça um treinamento+validação+teste usando biblioteca em Python para 1 ou
  mais modelo(s) inteligente(s) de sua escolha (redes neurais, árvores de decisão,
  ... ). Use estes valores como seu _baseline_ , seu resultado de referência. Lembre
  de armazenar o tempo de treinamento para todos os casos testados (neste e em
  próximos itens);
- Pruning
  **-** utilize estratégia (ou estratégias) de _pruning_ com grau de intensidade var-
  iável para reduzir a complexidade do(s) modelo(s);
  **-** compare os resultados de pruning e baseline em relação a: quantidade de
  parâmetros e taxa de acerto (idealmente fazendo um gráfico que relacione
  quantidade de parâmetros e taxa de acerto).
- Quantização
  **-** utilize estratégia (ou estratégias) de _quantização_ (pode ser para um modelo
  escolhido após o _pruning_ , para o modelo original ou para diversos modelos;
  **-** compare os resultados de quantização e _baseline_ em relação a: quantidade
  de bits e taxa de acerto para o problema (idealmente fazendo um gráfico
  que relacione número de bits e taxa de acerto).
- Avaliação em sistema embarcado
  **-** faça teste em microcontrolador que mostre o funcionamento e permita veri-
  ficar taxa de acerto e memória utilizada. Considere pelo menos 2 implemen-
  tações do sistema:

* com números em float para representar os parâmetros e entradas;
* com números inteiros para representar os parâmetros e entradas;

**-** obtenha respostas estimadas e compare com as implementações anteriores;
**-** calcule o valor mínimo, máximo, médio e a mediana do tempo de inferência
no sistema embarcado;

- Analise e discuta os resultados obtidos.
