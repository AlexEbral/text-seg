from text_gen_ru import generate

import time

ts = time.time()
generate(False)
print(time.time() - ts)