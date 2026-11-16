# Zadání projektu

Stav: dohodnutý směr před implementací. Datum: 16. 11. 2026.

## Účel

Vytvořit doplněk pro SD WebUI Forge Neo, který umožní lokálně převádět původní
modelové váhy stažené z Hugging Face v BF16, FP16 nebo FP32 do menších souborů
`.safetensors` použitelných v ComfyUI. Uživatel má rozumět tomu, co převádí,
jaké vrstvy zůstanou v původní přesnosti a jaká je ověřená kompatibilita výsledku.

## Identita

- Projekt: **Forge Neo Model Converter**.
- Repozitář / adresář doplňku: `sd-webui-forge-neo-model-converter`.
- Záložka ve Forge Neo: **Model Converter**.
- Popis: *Convert and quantize diffusion models for ComfyUI.*

## Dohodnutá rozhodnutí

1. Použít `comfy-kitchen` jako hlavní kvantizační knihovnu.
2. Vyčlenit samostatné převodní jádro a oddělit jej od integrace do Forge Neo.
3. Využít StarNodes Model Converter jako hlavní referenci a případně zdroj
   vybraných částí převodu a profilů po jejich ověření.
4. Kontrolovat export proti specifikaci, loaderu a oficiálním nástrojům ComfyUI.
5. Nevyžadovat `convert_to_quant` jako runtime závislost; může sloužit jako
   reference algoritmů a srovnávací implementace.
6. Podporovat volitelnou optimalizaci přímo s Prodigy+. Samostatně řešit algoritmus,
   ztrátovou funkci, finální zaokrouhlení a kontrolu přínosu optimalizace.
7. Původní kód projektu `sd-webui-convert-to-quant` není podkladem návrhu.
8. V této přípravné etapě napsat dokumentaci, nikoliv samotnou aplikaci. Nic nemazat.

## Rozsah vstupů

Základním vstupem jsou lokální původní `.safetensors` váhy. Podporovat postupně:

- jeden soubor checkpointu;
- shardovaný model s indexem a kontrolou úplnosti;
- HF snapshot / Diffusers strukturu pouze pro explicitně podporované adaptéry.

Oddělit podporu difuzního modelu, textového enkodéru, VAE a AIO checkpointu.
Přítomnost profilu jedné komponenty nesmí automaticky označit za podporované ostatní.
První ověřené modelové rodiny vybrat podle dostupných vzorků a skutečných potřeb
uživatele během prvního milníku. Seznam nyní není definitivně dohodnutý.

Kvantizace nemění automaticky Diffusers strukturu na ComfyUI state dict.
Kde je potřeba přemapování klíčů nebo rozdělení komponent, patří do modelového adaptéru.

## Cílové výstupy

| Uživatelská volba | Cílový kontrakt ComfyUI | Poznámka |
| --- | --- | --- |
| FP8 E4M3FN | `float8_e4m3fn` | Základní FP8 profil |
| FP8 E5M2 | `float8_e5m2` | Doplňková možnost po ověření |
| INT8 / INT8 ConvRot | `int8_tensorwise` | Scalar/per-row škály, volitelná rotace |
| INT4 ConvRot | `convrot_w4a4` | Explicitní pojmenování, ne obecné „INT4“ |
| W4A8 | `asym_w4a8_int8` | Packed 4bit váhy, dynamické INT8 aktivace |
| W6A8 | `w6a8_int8` | Packed 6bit váhy, dynamické INT8 aktivace |
| NVFP4 | `nvfp4` | Blokové škály a korektní padding/layout |
| MXFP8 | `mxfp8` | Mikroscaling a správný layout škál |

Zařazení v tabulce je cílový rozsah, nikoliv potvrzení dostupnosti v každé instalaci.
FP8, INT8 ConvRot a W4A8/W6A8 lze zahrnout do prvního vydání, pokud projdou
stejnými kompatibilitními kontrolami. Neodkládat W4A8/W6A8 pouze kvůli pořadí
formátů; rozsah řídit hlavně ověřenými modelovými rodinami.

Výstup může kombinovat kvantizované a původní vrstvy. Normy, biasy, embeddingy,
časové vstupy a další citlivé části ošetřovat podle profilu konkrétního modelu.
W8A16 lze později nabídnout jako volbu výpočtu nad INT8 uložením, nikoliv jako
nový fyzický formát. SVDQuant, AWQ a W3A8/W5A8 nejsou závazkem první verze.

## Uživatelský průchod

1. Vybrat vstupní soubor nebo podporovanou složku a samostatnou cestu výstupu.
2. Zobrazit rozpoznanou strukturu, modelový profil a dostupné formáty.
3. Zvolit formát, profil, zařízení a případnou optimalizaci.
4. Zobrazit plán bez zápisu: vrstvy, ponechanou přesnost, fallbacky a odhad velikosti.
5. Spustit převod v pracovním procesu; zobrazovat průběh a umožnit zrušení.
6. Zobrazit skutečné výsledky, upozornění, velikost souboru a validační report.

Odhad velikosti souboru neprezentovat jako odhad celkové VRAM nebo zrychlení inference.

## Nefunkční požadavky

- Primární prostředí: Windows a Forge Neo; jádro pokud možno přenositelné.
- Bez povinného běžícího ComfyUI a bez importů `comfy.*` v jádru.
- Ověření dostupných API Kitchen a podmínek cílového loaderu před převodem.
- Žádný automatický upgrade Torch/CUDA/Kitchen v prostředí hostitele.
- Postupné čtení, omezené pracovní buffery a návrh postupného zápisu výstupu.
- Zachovat vstupy; nekvantizovat nepozorovaně již kvantizovaný model podruhé.
- Viditelné fallbacky a chyby jednotlivých vrstev, žádné falešné hlášení úspěchu.
- Atomické dokončení výstupu a report s verzemi, profilem, parametry a výsledky kontrol.
- Volitelná optimalizace nesmí být podmínkou základního převodu.

## Akceptační kritéria první použitelné verze

- Záložka Model Converter ve Forge Neo spustí převod alespoň jedné zvolené
  modelové rodiny bez instalace celého ComfyUI.
- Plán převodu lze získat bez zápisu a předem odhalí nepodporované kombinace.
- Report uvede skutečné počty a typy převedených/ponechaných/selhaných vrstev.
- Výstup projde kontrolou tensorů, škál a per-module konfigurace `comfy_quant`.
- Každá kombinace označená jako ověřená projde cílovým loaderem ComfyUI
  a základním generováním; verze ComfyUI je v záznamu konkrétní.
- Převod je zrušitelný; zdroj zůstane nedotčen a neúplný výstup se nevydává za hotový.
- Velký syntetický vstup ověří paměťový limit podle velikosti největších vrstev
  a bufferů, ne pouze podle malé testovací sítě.
- Pro optimalizovaný režim je zaznamenána chyba až po finálním zakvantování;
  pokles pomocné ztráty se nevydává za prokázanou vizuální kvalitu.

## Mimo počáteční rozsah

Automatické stahování z HF a správa tokenů, univerzální podpora libovolné architektury,
trénování celých modelů, konverze libovolných pickle checkpointů, GGUF/K-quants,
merging/LoRA editor a plná náhrada všech StarNodes uzlů. Možná rozšíření nejsou
příslibem implementace bez dalšího zadání.
