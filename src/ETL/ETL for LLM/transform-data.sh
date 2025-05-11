#!binbash
set -eux

# Install required packages
pip install nltk

# Create output directory
mkdir -p processed

# Process data
python3 -  'PYCODE'
import os, json
raw_root = dataraw
out = open(dataprocessedall_meetings.jsonl,w,encoding=utf8)
for corpus in [ami,meetingbank]
    for root,_,files in os.walk(os.path.join(raw_root,corpus))
        for f in files
            if f.endswith((.txt,.trs,.json))
                try
                    path = os.path.join(root,f)
                    text = open(path,encoding=utf8,errors=ignore).read()
                    out.write(json.dumps({
                        corpus corpus,
                        file os.path.relpath(path, raw_root),
                        text text
                    }) + n)
                except
                    pass
out.close()
print(Wrote, os.path.getsize(dataprocessedall_meetings.jsonl), bytes)
PYCODE

# List the output
ls -lh processed