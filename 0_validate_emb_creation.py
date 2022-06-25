""" The embeddings should be saved in jsonl files, i.e. one line for each paper. 
 The following three files need to be encoded:
1. data/paper_metadata_mag_mesh.json
2. data/paper_metadata_view_cite_read.json
3. data/paper_metadata_recomm.json 

The embeddings must then reside in jsonl files with one json entry embedding per line, which will look something like this:
{"paper_id": "0dfb47e206c762d2f4caeb99fd9019ade78c2c98", "embedding": [-3, -6, 0, ..., 2]}

Number of lines in the jsonl files directly provided by specter embeddings - allennlp.
48473 cls.jsonl
36261 recomm.jsonl
142009 user-citation.jsonl

"""

import glob
import json

all_files = glob.glob("./data/*.json")
print(all_files)

for f in all_files:
    with open(f, 'r') as fin:
        data = json.load(fin)
        print(f, len(data))
        print(next(iter(data.items())))


# OUTPUT:
# ./data/paper_metadata_mag_mesh.json 48473
# ./data/paper_metadata_recomm.json 36261
# ./data/paper_metadata_view_cite_read.json 142009

# ./data/paper_metadata_mag_mesh.json 48473
# ('00014a8515491f0b3fe2a1ff6e0f5305e584dcd9', {'abstract': None, 'authors': ['115471397'], 'cited_by': [], 'paper_id': '00014a8515491f0b3fe2a1ff6e0f5305e584dcd9', 'references': [], 'title': 'ON THE CLASSIFICATION OF THE SCIENCES', 'year': None})
# ./data/paper_metadata_recomm.json 36261
# ('0002c0f45b3ef0f1491f91cbfefe9543e9af6163', {'abstract': 'In this paper we introduce an hp certified reduced basis method for parabolic partial differential equations. We invoke a POD (in time) / Greedy (in parameter) sampling procedure first in the initial partition of the parameter domain (h-refinement) and subsequently in the construction of reduced basis approximation spaces restricted to each parameter subdomain (p-refinement). We show that proper balance between additional POD modes and additional parameter values in the initial subdivision process guarantees convergence of the approach. We present numerical results for two model problems: linear convection-diffusion and quadratically nonlinear Boussinesq natural convection. The new procedure is significantly faster (respectively, more costly) in the reduced basis Online (respectively, Offline) stage.', 'authors': ['3122778', '39921175', '1905947', '2848614'], 'cited_by': ['16728d6f0a225bb8a71ebe4d0acd2512ca775327', '48d172aa8ee296b3a006c92933c5d13480d8e875', '0d912a4feb618c9c34ab5a1a8af16779ef7b3ac2', 'fc6aa7b07d9e4ca93dc66d338e76c1b1afebd06c', '2d3874361b849603553686f2840bdbdbfa5c2703', '7267485b5742b4d10f6aae3bab62244aea04f0bd', '1a180101c4987ab31da5460f929af841e1cf131c', '5398aa50d790126177ab1f6d735158d59776e20f', '61f408765886aeadf10636d97e7fd4bdc1ab17dc', '5dfc110a267f2d278c0e406d02cd8f34352ef5b1', '7dd499a7492ef95c7ee297c68eb0a88a66a47943', 'aebafc7ba7b8823ec278ffda3afe0305d93f85e9', 'd2c5baac729509be3e781746e474bc0a3807ff09', '13f43021859d54267a97c1f1a5f0f12041a05265', '8301686d0827e13c3a71925eba27c0dc51751f1e', '68e35fd7db55358d64bb41ee65f5df220b707533', '27aacb8070fab29c5afb49c27746612a2071a9bc', '0a9ff2ea85c2ffa5b502a8cff750312fb4c72c15', '3272d7c3e8406bc8d5c0039ba370026a42ad2b18', '8c877ac5c7d7e23b1d6bda5c5998109eeb73ac3c', '660b82654a3973688d517b4a60084c63511cbb6e', '1f14ba4f1fac5a2610e1519d2b9734357a51cb66', '98000d38467d839fa4c2fccbb25e58493cee5d36', 'b07681ad524695a5b22e378d164beffbe91e8edd', '10cfb9b4e5a275d257e0387b304572983a3e7756', '0386cfd2ccf5a8c778faa77ba62bd93d526d64c3', 'fa9a3889a5fc8a59c7b89d618782b2088b9fee51', 'e0a0ffb1ba82803e8a91adccd553913922493350', '60d3d4c0dbe10b691e5717350c656cfd0eb6a2d3', '32ea4589de0f23cbfe4b647c228f1e5ef8cd04c0', 'a9cebe5e3dd2785815b3b72b3dcac3dcc4b10dbb', 'd8c0ec46ffdc27fc125d241da80f93485375d2fc', '9298debbd1e37f924a9dd121a40b9725220375bc', '2936e52b2ed42a3c00b7648bb34fd9bc240663a7', '3d60b405e81203ebf039bec7c11ea98a11f86cd1', 'e0eda58c9c0ea0c1c79c086295da411bfcd7ede8', 'ef08f0a2ba05df4aa9ac484fcb5951ab56590098', '0779caa3862d3ae2837391d7d474fdce9412e4b6'], 'paper_id': '0002c0f45b3ef0f1491f91cbfefe9543e9af6163', 'references': ['07e1f620f68c0be579fb05bf6d231fa06b0db7c3', '7f2e6b565bc0cb164f8f21a3635914b032689fc6', '18dbc394176d2d768832a0a84c96168a57a462db', 'd0c1daf903c69a7098c82a0af69aef7cb5a797e7', 'c82e7782b4b902f51ba9982c1db47abb84ecb504', '245d0363a3a512faa775af835fe3af23b8c7f2b5', '660b82654a3973688d517b4a60084c63511cbb6e', '94b475d5c545058cf2b78fecb501b471090783fc', '2ee08c9979e0d94bd040982b0f7f66931abef177', '1170ed22fbe9dd6074417658c0190df1f2f3b0b2'], 'title': 'An hp Certified Reduced Basis Method for Parametrized Parabolic Partial Differential Equations', 'year': 2011})
# ./data/paper_metadata_view_cite_read.json 142009
# ('0003aa77bdefc1c75f9d2ba732635c132fc0c863', {'abstract': "PROBLEM STATEMENT\nPelvic girdle pain (PGP) is a common condition during or after pregnancy with pain and disability as most important symptoms. These symptoms have a wide range of clinical presentation. Most doctors perceive pregnancy related pelvic girdle pain (PPGP) as 'physiologic' or 'expected during pregnancy', where no treatment is needed. As such women with PPGP mostly experience little recognition. However, many scientific literature describes PPGP as being severe with considerable levels of pain and disability and socio-economic consequences in about 20% of the cases.\n\n\nOBJECTIVES\nWe aimed to (1) inform the gynecologist/obstetrician about the etiology, diagnosis, risk factors, and treatment options of PPGP and (2) to make a proposition for an adequate clinical care path.\n\n\nMETHODS\nA systematic search of electronic databases and a check of reference lists for recent researches about the diagnosis, etiology, risk factors and treatment of PPGP.\n\n\nRESULTS\nAdequate treatment is based on classification in subgroups according to the different etiologic factors. The various diagnostic tests can help to make a differentiation in the several pelvic girdle pain syndromes and possibly reveal the underlying biomechanical problem. This classification can guide appropriate multidimensional and multidisciplinary management. A proposal for a clinical care path starts with recognition of gynecologist and midwife for this disorder. Both care takers can make a preliminary diagnosis of PPGP and should refer to a physiatrist, who can make a definite diagnosis. Together with a physiotherapist, the latter can determine an individual tailored exercise program based on the influencing bio-psycho-social factors.", 'authors': ['40572137', '3675075', '48815127'], 'cited_by': ['e7e7a7fc07f516fd39b4b0cb7ff3a2acfe837c1a', 'e87c5465ae8d9e22c237b8a1735ab5992466f1c4', 'c35d73aed8ccf2380988df0e7f33a16352c3250e', 'd6d34353052938630c216bb502958e0c5fa8151f', 'a3848ea423dd9d5ac2409df2444b63a323908191', '29552e98facc811dd37096f95fde28ab8c3a350f', '7814552e978d3ebda7abd56aaeff6d7f6567df93'], 'paper_id': '0003aa77bdefc1c75f9d2ba732635c132fc0c863', 'references': ['420604a2d0161cd5b5d2df75dd6252f224c8b055', 'e78149afc38dac0f78bad1b8020f24cf08dcb5f7', 'b613facdfc71d82c1a78b5b652b239aed9be42ff', 'c9c4cb05b9e69907544c59c7fe431860195c9f1c', '5b5a6f447e000fe70eadc505a746a9b2b396a401', '6e70287fc5b78938a78abaae9e4f3c7f7b343560', 'ee1bd624aa3df264a5b71f81528f8fe64d05471d', '0137dcff4d46e31f4bdcf1ce1a0ee2ba1f80abff', '58d712de2f8ed57ebd4697eb49c341c1fc113287', '5620994a1e6420d9119002a26e5792f6043b87f4'], 'title': 'Pelvic Girdle Pain during or after Pregnancy: a review of recent evidence and a clinical care path proposal', 'year': 2013})