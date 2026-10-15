# Guia de sound design: Podcast Plus (true crime)

Este guia é lido pelo Claude sempre que cria o sound design de uma cena. Podes editá-lo na página
Estrutura Plus. Quanto mais concreto, melhor o resultado.

## 1. Objetivo: pintar a cena com som

O ouvinte tem de "ver" a cena com os ouvidos. Uma cena bem desenhada tem quatro camadas:

1. **Ambiente** (sempre presente): o lugar e a hora. Sala vazia, rua à noite, corredor de tribunal, chuva
   contra a janela, trânsito distante, frigorífico a zumbir, vento num descampado. É a base e quase nunca
   desaparece. Entra no início da cena e muda quando o lugar muda.
2. **Foley** (ações pequenas e constantes): passos, roupa, chaves, papéis, copos, cadeiras, respiração,
   isqueiro, cigarro, caneta, gavetas, sacos de plástico, botões de elevador. É o que dá realismo.
3. **Efeitos de ação** (o que a narração descreve): portas, carros, telefones, tiros, sirenes, algemas,
   fechaduras, um comboio, uma ambulância.
4. **Pontuação dramática** (poucos e discretos): um impacto grave, um drone que sobe, um stinger seco,
   um silêncio súbito. Só nas viragens (uma revelação, uma descoberta, uma mentira desmascarada).

Entre cenas usa uma **transição** curta (riser suave, whoosh grave, cauda de reverb) no fim da cena
ou no início da seguinte.

## 2. Densidade

Não deixes a cena "despida". Como regra de partida:

- Há ambiente por baixo da cena quase toda.
- Uma frase de ação ou de descrição física pede 1 a 3 sons (foley e efeitos).
- Uma frase de reflexão, opinião ou análise pede só ambiente, música ou silêncio.
- Procura 1 elemento de som por cada 5 a 8 segundos de narração, para além do ambiente.
- Sons seguidos do mesmo tipo (passos, relógio) podem repetir-se se a ação se prolonga, mas varia a descrição.

## 3. Ler o texto literalmente (regras críticas)

Põe sons apenas de coisas que **acontecem de facto, no momento descrito**.

- **Negações:** "os polícias não bateram à porta", "ninguém ouviu nada", "sem um som", "não tocou" →
  NÃO ponhas o som negado. Se a negação é o ponto dramático, usa ambiente tenso ou silêncio.
- **Hipóteses e condicional:** "se tivesse tocado", "podia ter gritado", "teria entrado" → não.
- **Perguntas e dúvidas:** "Quem teria ouvido o tiro?" → não ponhas o tiro.
- **Metáforas e expressões:** "o mundo desabou", "caiu-lhe o chão", "bateu-lhe o coração" usados em
  sentido figurado → não ponhas desabamentos, quedas ou batimentos literais.
- **Factos de contexto:** "em 1994 ouviram-se tiros no prédio", contados de passagem, não são sons da
  cena presente, a menos que a narração a esteja a reencenar. Quando o narrador conta uma cena do passado em
  tempo presente ("Entra. Fecha a porta."), aí sim, reencena com sons.
- **Verbos de fala** (disse, perguntou, gritou) não são sons: não ponhas vozes nem diálogo.
- **Quantidades e ordem:** respeita-as ("bateu duas vezes", "três tiros").
- **Quando estiver em dúvida, não ponhas o som.** Um som a mais num sítio errado estraga a cena; um som a
  menos quase ninguém repara.

## 4. Exemplos

**Negação**
Texto: "Os polícias não bateram à porta. Entraram."
Certo: sem batidas. Porta aberta com força, passos pesados de várias pessoas, ambiente tenso.
Errado: batidas à porta.

**Metáfora**
Texto: "Quando ouviu a notícia, o mundo desabou."
Certo: ambiente, talvez um drone grave e um silêncio. Errado: efeito de desabamento.

**Hipótese**
Texto: "Se a vizinha tivesse acordado, teria ouvido os gritos."
Certo: ambiente de noite, silêncio. Errado: gritos.

**Cena densa (exemplo de boa pintura)**
Texto: "Chegou a casa às duas da manhã. Pousou as chaves na mesa, tirou o casaco e serviu-se de um copo de água."
Certo: ambiente de casa à noite (zumbido do frigorífico, rua distante), carro a estacionar ao longe, porta de entrada
a fechar, chaves a pousar, roupa/casaco a ser tirado, água a ser vertida num copo.

## 5. Estilo sonoro: true crime

- Realista, frio, desconfortável, contido. Nada de efeitos de filme de ação nem cómicos.
- Ambientes baixos e naturais; dá espaço à voz. Sons curtos e secos funcionam melhor do que longos e cheios.
- **Música:** só quando a cena pede e sempre em segundo plano. Drones, piano esparso, cordas graves,
  texturas, sem melodia marcante nem percussão forte. Muitas cenas ficam melhor só com ambiente.
- **Evita:** sirenes constantes, música de suspense óbvia, batidas de coração, relógios a fazer tic-tac em tudo.

## 6. Como descrever cada som ao gerador (prompts em inglês)

- Concreto e físico: material, distância, espaço acústico e duração aproximada.
  Bom: "Heavy wooden apartment door being pushed open, creaking hinges, in a quiet stairwell".
  Mau: "door sound".
- Indica a distância quando importa ("distant", "close"), e o espaço ("empty room", "outdoors at night").
- Para ambiente: "seamless", sem música e sem vozes.
- Não peças vozes humanas a falar. Murmúrio indistinto de multidão é aceitável.

## 7. Volumes (campo gain, de 0.1 a 1)

- Ambiente e música: 0.2 a 0.5.
- Foley: 0.4 a 0.7.
- Efeitos de ação em primeiro plano: 0.6 a 1.
- Pontuação dramática: 0.5 a 0.8.
