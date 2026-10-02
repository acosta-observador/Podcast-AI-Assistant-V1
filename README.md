# Podcast Plus AI Assistant

Plataforma web de apoio aos sonoplastas dos podcasts Plus do Observador. Primeiro módulo: **AutoTake**.

## O que faz o AutoTake

Recebe a gravação completa e o guião do episódio. Para cada bloco (parágrafo) do guião encontra todas as tentativas do narrador, escolhe a melhor e devolve:

- uma grelha de revisão no site, com todos os takes de cada bloco, o escolhido destacado e troca com um clique ou com o teclado;
- um projeto **Reaper** (.rpp) com a montagem, as alternativas em pistas silenciadas por baixo e uma região por bloco (os que precisam de revisão começam por "REVER");
- o **áudio montado** em WAV 24-bit, com a taxa de amostragem e os canais do original.

### Regras de escolha

1. Por defeito ganha o **último take completo** do bloco.
2. Se depois dele o narrador regravou só uma parte (retoma), o AutoTake monta o take completo com a retoma e marca para confirmar o corte.
3. Se nenhum take está completo, propõe uma **montagem composta** frase a frase (coluna C na grelha).
4. Tudo o que não tem confiança alta fica com um ponto amarelo para o sonoplasta ouvir.

### Pipeline (`autotake/`)

| Ficheiro | Etapa |
|---|---|
| `audio.py` | Cópia FLAC 16 kHz mono para transcrever, MP3 para ouvir no browser, envelope de energia, cortes e montagem com ffmpeg |
| `transcribe.py` | ElevenLabs Scribe (`scribe_v2`, português, timestamps por palavra), com cache em JSON |
| `script_parser.py` | Guião `.docx`/`.txt` em blocos e frases; ignora indicações entre parênteses retos e linhas curtas em maiúsculas |
| `align.py` | Procura cada frase na transcrição (sementes + alinhamento LCS com rapidfuzz); deteta recomeços a meio da frase |
| `takes.py` | Agrupa ocorrências em takes, calcula cobertura, palavras fora do guião, pausas longas e escolhe |
| `export.py` | Projeto Reaper |

Os pontos de corte procuram o instante de menor energia entre a palavra anterior e a primeira palavra do take (e o mesmo no fim), para não cortar respirações nem palavras.

## Correr localmente

Precisa de Python 3.11+ e ffmpeg.

```bash
pip install -r requirements.txt
export ELEVENLABS_API_KEY=...        # a chave do Scribe
export APP_PASSWORD=...              # opcional
uvicorn app.main:app --reload
```

Abrir http://localhost:8000.

Ou com Docker:

```bash
docker build -t podcast-plus-ai .
docker run -p 8000:8000 -v $(pwd)/data:/data -e ELEVENLABS_API_KEY=... -e APP_PASSWORD=... podcast-plus-ai
```

## Pôr online (Render)

1. Criar um repositório no GitHub e carregar estes ficheiros.
2. No Render: New → Blueprint → escolher o repositório. O `render.yaml` cria o serviço com disco persistente.
3. Quando pedir, preencher `ELEVENLABS_API_KEY` e `APP_PASSWORD`.

## Pôr online (genérico)

A app é um único container com disco persistente. Funciona em Railway, Render, Fly.io ou numa VM do Observador:

1. Criar o serviço a partir do repositório (usa o `Dockerfile`).
2. Montar um **volume persistente** em `/data` (é onde ficam gravações, transcrições e escolhas).
3. Definir `ELEVENLABS_API_KEY` e `APP_PASSWORD`.
4. Confirmar que o proxy da plataforma aceita uploads grandes (gravações de 1h em WAV passam de 1 GB). Se não aceitar, enviar FLAC ou MP3 de alta qualidade.

## Afinar o algoritmo

Os limiares estão no topo de `align.py` e `takes.py`. Para testar sem gastar Scribe, depois do primeiro processamento a transcrição fica em `data/episodes/<id>/work/transcript_scribe.json`:

```bash
python cli.py data/episodes/<id>             # volta a alinhar e mostra as decisões
```

No site, o endpoint `POST /api/episodes/<id>/reprocess` faz o mesmo.

**Validação recomendada:** 2 ou 3 episódios antigos com raw, guião e versão final do sonoplasta. Medir em quantos blocos o AutoTake escolhe o mesmo take.

## Limitações desta versão

- Uma pista (narrador). Multipista ainda não.
- Um episódio processado de cada vez (`WORKERS=1`); a fila vive no processo, e jobs interrompidos por um reinício retomam sozinhos.
- Guardar em disco local; passar para R2/S3 quando houver mais utilizadores.
- Exporta só Reaper; EDL/marcadores para Pro Tools e Audition ficam para a próxima.

## Próximos módulos

Cada módulo usa o mesmo episódio (pasta, áudio, guião). Candidatos: limpeza de ruído, normalização de loudness, deteção de respirações e cliques.
