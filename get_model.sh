#!/bin/bash

FILE=models/fasttext/araneum_none_fasttextcbow_300_5_2018.model

if [ ! -f "$FILE" ]; then
  wget https://rusvectores.org/static/models/rusvectores4/fasttext/araneum_none_fasttextcbow_300_5_2018.tgz
  tar -xvzf araneum_none_fasttextcbow_300_5_2018.tgz -C models/fasttext
  rm araneum_none_fasttextcbow_300_5_2018.tgz
fi

echo "Model is ready"