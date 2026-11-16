# Zdroje a závěry průzkumu

Datum posouzení: 16. 11. 2026. Podklad: předchozí diskuse a inspekce upstream
README a relevantního zdrojového kódu. Neproběhl benchmark ani generování.
Závěry je nutné při implementaci znovu ověřit proti konkrétním verzím.

## Primární zdroje

| Zdroj | Úloha v projektu |
| --- | --- |
| https://github.com/Comfy-Org/comfy-kitchen | Hlavní kvantizéry, layouty, serializace tensorů a runtime backendy |
| https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md | Kontrakt exportovaných checkpointů |
| https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ops.py | Skutečná implementace loaderu a operací |
| https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/quant_ops.py | Registr formátů |
| https://github.com/Comfy-Org/comfy-model-tools/blob/main/quant_int8_convrot.py | Oficiální reference exportu INT8 ConvRot, W4A8 a W6A8 |
| https://github.com/Starnodes2024/comfyui-starnodes-modelconverter | Reference celého převodníku a modelových profilů |
| https://github.com/LoganBooker/prodigy-plus-schedule-free | Volitelný optimalizátor Prodigy+ |
| https://github.com/silveroxides/convert_to_quant | Reference learned rounding, profilů a srovnávacího převodu |
| https://github.com/Comfy-Org/comfy-quants | Pokročilé offline a modelově specifické exportní postupy |
| https://github.com/Haoming02/sd-webui-forge-classic/tree/neo | Hostitelské prostředí Forge Neo |

Ověřené snapshoty části zdrojů z úvodního průzkumu:

- Kitchen: `5beaba69afa63b71e86677c49093f04d621b07f1`.
- ComfyUI: `e9afbccc2669677076feb08e52658f129cd8c12b`.
- comfy-model-tools: `d6797787e6bdb1a1fb0094d588a26f8e71a1c757`.
- CTQ: `a45e94527e3e2762e616e948bfd773d8a020c55d`.
- Forge Neo: `14223a4b84e009d5cf0132a2245ea17628ca479d`.

Tyto hashe nejsou instalační piny ani prohlášení kompatibility. Snapshoty StarNodes
 a Prodigy+ zaznamenat při prvním převzetí kódu; nevymýšlet je zpětně.

## Rozhodnutí a důvody

### Kitchen jako hlavní základ

Nabízí stavební prvky pro většinu požadovaných formátů a Forge Neo ji již instaluje.
V prověřeném requirements větve neo byla připnutá verze `0.2.41`. Samotná Kitchen
nenahrazuje rozpoznávání vstupu, profily, bezpečný export ani UI. Doplněk musí
respektovat konkrétní verzi hostitele a ověřovat požadované funkce.

### StarNodes jako reference a zdroj vybraných částí

`star_ultimate_converter.py` obsahuje převodní funkce, práci se shardy, profily,
exportní konfigurace i zjednodušený learned rounding. Importy `folder_paths`
a `comfy.utils` potvrzují vazbu na ComfyUI; Pro varianta také načítá výstup přes
`comfy.sd`. Tyto vazby oddělit od nového jádra.

Načítání sestavuje celý vstupní slovník a vytváří výstupní slovník; přímý port
neřeší požadavek omezené RAM. Převzít koncepty i vhodné části po auditu,
ne celý uzel jako povinnou závislost. Repozitář obsahuje licenci MIT;
při převzetí zachovat oznámení a evidovat upstream commit.

### Prodigy+ přímo, bez povinného CTQ

Prodigy+ aktualizuje parametry podle gradientů. Nedodává kvantizační algoritmus,
loss, data ani export. CTQ poskytuje další logiku nad optimalizátorem;
StarNodes již obsahuje zjednodušený port části této logiky s AdamW.

V prověřených větvích CTQ se `ProdigyPlusScheduleFree` vytváří s
`use_schedulefree=False`. Pouhé použití výchozí konfigurace Prodigy+ proto
nereprodukuje CTQ. Prodigy+ má Apache-2.0 licenci; verzi a požadavky
konkrétní převzaté části ověřit před distribucí.

### Co nedokazuje README

Seznam formátů v UI není důkaz kompatibility všech variant s běžným ComfyUI.
StarNodes cesta označená `svdquant_w4a4` v prohlédnutém kódu zapisovala
`format: convrot_w4a4`, příznak `svdquant` a tensory `svdq_l1` / `svdq_l2`.
V prověřeném `comfy/ops.py` nebyla nalezena jejich obsluha. Jde o otevřenou otázku,
ne o důkaz kompatibility ani úplný audit celého runtime. Nepřevzít ji jako
podporovaný standardní SVDQuant bez kontroly cílové implementace.

W3A8/W5A8 podobně vyžadují odpovídající kvantizér i registraci loaderu.
Export, načtení a zrychlený výpočet jsou tři samostatné schopnosti.

## Exportní zásady

- Zapisovat per-module `<module>.comfy_quant` jako uint8 UTF-8 JSON.
- Zachovat názvy tensorů podle cílové architektury a prefixy komponent.
- Packed weights a škály vytvářet pomocí vhodných Kitchen layoutů.
- Nepoužívat resident-only optimalizace jako uložitelný canonical layout.
- Pro W4A8/W6A8 loader nepřebírá volitelný `weight_correction`; používat
  kompatibilní symetrické váhy.
- `full_precision_matrix_mult` ovlivňuje výpočet, ne bitovou šířku uložených vah;
  neznamená automaticky výpočet FP32.
- Nedeklarovat univerzální velikost souboru ani zrychlení podle počtu bitů.

## Otevřené otázky pro první implementační milník

1. První modelové rodiny, komponenty a vzorky pro ověření generováním.
2. Minimální podporované verze Forge Neo, Kitchen, Torch a ComfyUI.
3. Vybrané části a konkrétní commity StarNodes pro převzetí.
4. Kompatibilní metoda postupného zápisu jednoho safetensors souboru a rozpočet RAM/disku.
5. Rozsah mapování původních HF/Diffusers klíčů do ComfyUI pro první adaptéry.
6. Metrika, algoritmus a počáteční formáty volitelné optimalizace.
7. První ověřené CPU/GPU backendy a dostupnost testovacího hardware.
8. Licence vlastního projektu; v přípravné etapě nebyla vybrána.
