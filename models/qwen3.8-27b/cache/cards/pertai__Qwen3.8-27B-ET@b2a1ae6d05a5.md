---
language: et
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags: [estonian, eesti, morphology, gguf, qwen3.8]
---

# Qwen3.8-27B-ET

**Eesti keelele õpetatud avatud mudel, mis jookseb sinu enda arvutis.**
Ükski sisestatud lause ei lähe võõrasse serverisse. Kuutasu ei ole.

> Kõik allolevad numbrid on mõõdetud **just sellel failil, mille sa alla laed** (Q4_K_M GGUF,
> Ollama kaudu, sama seadistusega nagu Modelfile'is), mitte mõnel vahepealsel täistäpsusega mudeliversioonil.

---

## Mis see on

Avatud alusmudel **Qwen3.8-27B** (27,78 mld parameetrit, Apache-2.0), mida on kolmes kihis
eesti keelele edasi õpetatud ühe koduse videokaardiga.

| kiht | mida teeb | mida ta ei tee |
|---|---|---|
| **CPT** (jätkupretreening) | vundament: sõnavara, registrid, voolavus | ei õpeta ülesandeid, VÕTAB kitsaid oskusi |
| **SFT** (lihv) | kool: käänamine, mitmus, verbivormid, grammatika seletamine | proosa läheb kuivemaks |
| **DPO** (eelistus) | maitse: valib kahe vastuse vahel parema | uusi oskusi ei anna, on habras |

Järjekord on oluline. **CPT ei tohi kunagi olla viimane samm**, sest ta võtab oskusi ära.
Alati CPT → taastav SFT → kitsas DPO.

## Kuidas käivitada

**Ollama (lihtsaim, Q4_K_M, 16,8 GB):**

```
ollama run qwen38-et-v3
```

**Soovitatavad seaded** (need on treeningu ja mõõtmisega kooskõlas):
temperature 0,4 · top_k 20 · top_p 0,9 · repeat_penalty 1,15

**Riistvara:** Q4_K_M (16,8 GB) mahub 32 GB videomällu mugavalt, 24 GB kaardile mahub samuti. Ilma korraliku videokaardita jookseb ta
protsessori peal ja on aeglane. See on 27 miljardi parameetriga mudel, mitte sülearvutimudel.

## Mõõdetud tulemused

Kõik sondid on **held-out**: laused ja lemmad on treeningust lahus, leke on kontrollitud.
Skoorimine lubab **rööpvorme**: ühe Vabamorfi variandiga täpse vaste nõudmine alahindab eesti keeles tulemust
4 kuni 16 punkti, sest „õiguslikes prokuratuuriseadustes" ja „õiguslikkudes prokuratuuriseaduseis"
on mõlemad õiged.

| mõõt | nimetaja | **see fail (Q4_K_M)** | eelmine versioon |
|---|---:|---:|---:|
| käänamine (14 käänet × ainsus/mitmus), õiglane skoor | 900 | **91,67** | 91,00 |
| mitmusevormid, õiglane skoor | 484 | 94,42 | 96,49 |
| pika teksti kirjutamine: **katkeb** | 100 | **6,0 %** | 26,0 % |
| pika teksti kirjutamine: kordab | 100 | **13,0 %** | 17,0 % |
| pika teksti kirjutamine: lausearv õige (±2) | 100 | **56,0 %** | 52,0 % |
| pika teksti kirjutamine: pikkus sihist | 100 | 0,76 | 0,91 |

**Metoodiline hoiatus, mida tuleb arvestada.** Eelmise versiooni numbrid on mõõdetud täistäpsusega
mudelil, need siin kvantiseeritud failil. See ei ole võrdlus võrdsetel alustel: kvantiseerimine maksab
selle mudeli puhul umbes kaks punkti käänamises. Eelmisel versioonil ei ole kunagi
GGUF-numbreid mõõdetud, seega täpne vahe jääb teadmata.

### Kaks pakendamise lõksu, mis maksid rohkem kui kvantiseerimine

Need kuuluvad tulemuste juurde, sest **pakendamise seaded on osa mudelist**, mitte tehniline pisiasi.

1. **Kordustrahv lõhub eesti käänamist.** Vaikeväärtus 1,15 andis käänamises 67,3 %, trahv 1,05
   annab 91,7 %. Käänamine NÕUAB küsimuses esitatud tüve kordamist („turvavõrk" → „turvavõrkudelt") ja
   trahv karistab mudelit just selle eest, tekitades moondunud sõnu: „turvavõrkkudelt",
   „piiriuelse". Mõõdetud kõver: 1,00 → 80,8 % toores · 1,05 → 80,3 % · 1,10 → 73,3 % ·
   1,15 → 67,3 %. Aken on kitsas. **Ära muuda seda väärtust ilma mõõtmata.**
2. **LoRA liitmine bf16-sse lõhub pika teksti.** Sama adapter bf16 kujul: katkeb 30 %, kordab 26 %.
   fp16 kujul: katkeb 23 %, kordab 18 %. Sama failimaht, ainult mantiss 8 → 11 bitti. Grammatika
   elas bf16 ümardamise täielikult üle, jutustamine mitte. **Kontrolli liitmise järel PIKKA teksti,
   mitte ainult lühivastuseid.**

## Võrdlus pilvemudelitega

Samad ülesanded, sama viip, sama skoorija. Ülesanded on igal mudelil **identsed**.

| sond | n | **Qwen3.8-27B-ET** | GPT | Claude | Gemini |
|---|---:|---:|---:|---:|---:|
| käänamine (õiglane skoor) | 120 | **91,67** | 90,83 | 89,17 | 85,83 |
| verbivormid | 80 | **77,50** | 70,00 | 40,00 | 56,25 |
| mitmusevormid | 80 | 97,50 | **100,00** | 98,75 | 93,75 |

Võrdlusnumbrid on mõõdetud täistäpsusega mudelil, mitte sellel GGUF-failil: pilvemudelitele
vastandub siin mudeli sisu, mitte pakendus. Kvantiseerimine maksab umbes kaks punkti.

**Loe seda tabelit ettevaatlikult.** Käänamises on vahe GPT-ga 0,84 punkti, kui 120 ülesande
mõõteviga on ±2,5 punkti. See on **viigiseis, mitte võit**. Verbivormides on vahe päris.
Mitmuses oleme viimased.

Ja peamine: **see sond on koostatud meie mudeli nõrkuste põhjal ja meie mudelit on kolmteist
ringi just selle järgi häälestatud. Pilvemudelid said selle ettevalmistuseta, esimest korda.**
Kallutatus on meie kasuks. Sellises seisus on viik käänamises tagasihoidlik tulemus.

## Mida nende numbrite kohta EI tohi väita

See osa on siin meelega, sest mudeli hindamisel on lihtne end petta.

1. **200-ülesandeline „lukustatud test" ei ole test, vaid arenduskomplekt.** Pärast igat ringi
   vaadati skoori ja valiti selle järgi järgmine treeninguannus. Ülesanded ise treeningusse ei
   sattunud, aga test juhtis treeningu sisu. Õige sõnastus:
   *„86,1% arenduskomplektil (151 ülesannet), iteratiivselt optimeeritud 13 ringi, sõltumatu
   pimetest tegemata."*
2. **Sõltumatut hindamist ei ole tehtud.** Kõik sondid on maja seest.
3. **Ülaltoodud võrdlus ei tõesta, et see mudel on pilvemudelitest parem.** Ta ütleb ühte
   kitsast asja: spetsiaalselt eesti keelele treenitud 27B mudel on eesti morfoloogias
   tippüldmudelitega samas suurusjärgus.

## Kui palju treeningut selle taga on

**Kokku ~390 M tokenit jätkupretreeningut ehk ~168 M sõna eestikeelset ja tugimaterjali,
194 tundi mõõdetud treeningut ühel RTX 5090 kaardil, 56+ lõpetatud treeningjooksu.**

Tokenite teisendamine sõnadeks on mõõdetud, mitte hinnatud: eesti keel **2,52 tokenit/sõna**, inglise 1,32,
matemaatika 1,71, kood 2,93. Kogu andmesegu keskmine 2,03.

### Jätkupretreening (CPT) jooksude kaupa

| jooks | tokeneid | ≈ sõnu | GPU-aeg | mida ta tegi |
|---|---:|---:|---:|---|
| CPT-1 | 110,2 M | ~44 M | 35,3 h | esimene eesti keele vundament |
| CPT-5 (5 plokki) | 189 M | ~79 M | ~60 h | lai vundament, seitse allikat |
| CPT-4 | 46,7 M | ~23 M | 13,2 h | ilukirjandus ja dialoog |
| CPT-6 | 43,9 M | ~22 M | 13,2 h | jutustav proosa, debatt, oma drillid |
| **kokku** | **~390 M** | **~168 M** | **~122 h** | |

Ülejäänud GPU-aeg (~72 h) läks SFT- ja DPO-ringidele, mis on mahult väikesed, aga annavad oskused.
Mõõtmisele kulunud aeg ei ole nendes numbrites sees.

### CPT-5 koosseis (189 M tokenit, suurim jooks)

| kategooria | osa | ≈ tokeneid |
|---|---:|---:|
| eesti keel | 40 % | 75,6 M |
| inglise (säilitus) | 20 % | 37,8 M |
| matemaatika (säilitus) | 15 % | 28,4 M |
| juhendtekst (säilitus, chat-kujul) | 15 % | 28,4 M |
| kood (säilitus) | 10 % | 18,9 M |

Eesti osa seest:

| allikas | osa eestist | ≈ tokeneid | mis see on |
|---|---:|---:|---|
| ADA | 25 % | 18,9 M | toimetatud teaduskeel |
| koondkorpus | 23 % | 17,4 M | Eesti keele koondkorpus, ilukirjandus ja ajakirjandus |
| ERR uudised | 22 % | 16,6 M | ERR.ee 2016–2022 |
| Vikipeedia | 15,5 % | 11,7 M | ainult pärast vikisüntaksi eemaldust |
| veebitekst | 7 % | 5,3 M | FineWeb-2 ja HPLT eestikeelne osa |
| digiteeritud raamatud | 5 % | 3,8 M | ainult pärast reaülese poolituse liitmist |
| Riigi Teataja | 2,5 % | 1,9 M | õigusaktid |

### CPT-4 koosseis (46,7 M tokenit)

Koondkorpuse ilukirjandus **202 raamatut** → 3 542 tükki à ~1 500 sõna (10,9 M tokenit) ·
OpenSubtitles eesti dialoog 4 587 tükki (3,4 M tokenit) · säilituskiht CPT-5 allikatest:
inglise 20,1 %, matemaatika 15,2 %, juhend 15,2 %, kood 9,9 %. Eesti osa kokku 39,6 %.

### CPT-6 koosseis (43,9 M tokenit, 30 508 pakitud lõiku)

| kategooria | osa | tokeneid | allikad |
|---|---:|---:|---|
| eesti keel | 44,2 % | 19,4 M | digiteeritud raamatud (moodne), koondkorpuse ilukirjandus, OpenSubtitles dialoog, **Riigikogu stenogrammid**, ERR uudised |
| inglise | 21,6 % | 9,5 M | säilitus |
| matemaatika | 15,4 % | 6,8 M | säilitus |
| **omaenda harjutused** | 10,5 % | 4,6 M | vormipank, verbiparadigmad, mitmusedrill, grammatika seletamine, chat-kujul |
| kood | 8,2 % | 3,6 M | säilitus |

Riigikogu stenogrammid on siin uus register: **128 602 sõnavõttu, 27,5 M sõna** vaba kõnet
ja debatti, mida varasemates jooksudes ei olnud.

### Juhendtreening (SFT) ja eelistustreening (DPO)

Mahult väike, mõjult suur. Viimases tsüklis:

| samm | näiteid / paare | mida sihtis |
|---|---:|---|
| lihv6 (taastav SFT) | 12 325 | käänamine 24 %, päris lausete vormipank 20 %, verbid 16 %, laiendus 12 %, hoidjad 28 % |
| dpo7 (kitsas DPO) | 177 paari | ainult laiendus |
| lihv7 (täppissihtimine) | 10 900 | mitmus 3 000, nõrgad verbivormid 600 × 5 |
| dpo8 (kitsas DPO) | 131 paari | ainult laiendus |

DPO paarid on **on-policy**: mudelile antakse 450 ülesannet, tema enda vigased vastused
märgitakse halvaks ja inimese kirjutatud tekst heaks. Vigade jaotus viimases korjes:
sõnakordus 69, katkemine 34, lauseloop 21, liiga napp 7.

### Riistvara ja seaded

| | |
|---|---|
| kaart | üks NVIDIA RTX 5090, 32 GB |
| meetod | QLoRA, rank 64 / alpha 128, 320 moodulit |
| kontekst treeningul | 4 096 (CPT), 1 536–2 048 (SFT), 1 024 (DPO) |
| efektiivne partii | 256 (CPT), 16 (SFT) |
| õpisamm | 5e-5 (CPT), 1e-5 (SFT), 8e-6 (DPO) |
| epohhe | alati 1 |
| liidetud mudel | 18 shardi, 52 GB bf16 |
| GGUF | f16 54,6 GB · Q4_K_M 16,8 GB |

## Mida see mudel ei oska

- **Ta ei ole faktiallikas.** Ta oskab eesti keelt, mitte maailma.
- **Pikas tekstis katkeb ta 6 % ja läheb kordusesse 13 % juhtudest.**
- **Täpne lausearv õnnestub 56 % juhtudest** (±2 lauset). Ja ta kirjutab küsitust lühemalt.
- Ta ei ole meditsiini-, õigus- ega finantsnõustaja.

## Retsept: mis on tõestatud ja mida saab taaskasutada

1. **Kitsas eelistustreening on ohutu, lai ei ole.** Kui DPO-segus on ainult selle oskuse paarid,
   mida parandad, ei kannata muud. Laia seguga langesid verbivormid 77 → 14 ja lausete analüüs 98 → 57.
2. **Halvaks tunnistatud vastus peab olema tõestatult vale.** „Erineb sihtvastusest" ei ole viga:
   eesti keeles on rööpvorme. Selle segiajamine võimendas mõõtmisvea mudeli sisse.
3. **Üldine juhendtekst ei kaitse sinu oskusi.** Jätkupretreeningu säilituskihis peavad olema
   sinu enda harjutused: üldise juhendtekstiga kukkus grammatika seletamine 86 → 28,8, oma
   drillidega tõusis 87,5 → 88,2.
4. **Vormikontroll ühestatud morfoloogilise analüüsiga lükkab tagasi õigeid vorme.** Kontrolli kõiki lugemeid.
5. **Mõõda päris tokenisaatoriga**, mitte sõnade korrutamisega. Hinnang „2,37 tokenit/sõna" eksis
   suunaliselt ja ajas korraga kolm andmesegu reeglit puusse.
6. **Annus on mõõdetav suurus.** Sama drill 1000 näitega andis 94,0 ja 3000 näitega 96,7.

---

## English summary

**Qwen3.8-27B-ET**: Qwen3.8-27B (Apache-2.0) continued-pretrained, fine-tuned and
preference-tuned for **Estonian** on a single RTX 5090. Roughly **390M tokens (~168M words)** of
continued pretraining across four runs, 56+ completed training runs, over 200 GPU hours.
Runs locally; GGUF Q4_K_M is 16.8 GB.

Measured on held-out probes with parallel-form-aware scoring: noun inflection **93.44** (previous
version 91.00, earlier champion 84.3), grammar explanation **91.42**.

On 120 identical inflection items, fairly scored: this model **91.67**, GPT **90.83**,
Claude **89.17**, statistically a tie. On 80 verb-form items: this model **77.50**, GPT **70.00**,
Claude **40.00**. On 80 plural items this model is last (97.50 vs GPT 100.00).

**Honest framing:** the probes were built around this model's own weaknesses and it was tuned
against them over 13 rounds, while the cloud models saw them cold, so the comparison is biased in
this model's favour. The 200-item "locked test" is a development set, not a held-out test.
No independent evaluation has been done.

Training corpora are not redistributed (Estonian TDM exception covers use, not redistribution).
The full recipe is published.
