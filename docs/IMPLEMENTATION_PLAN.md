# Architektura a plán implementace

Datum: 16. 11. 2026. Stav: plán, žádná implementační etapa zatím není zahájená.
Přípravná etapa: nový T3 projekt, soukromý GitHub repozitář a tato dokumentace.

## Navržené členění

Následující názvy představují plán; příslušné moduly se v této etapě nevytvářejí.

- `scripts/`: tenká integrace Gradio a životního cyklu Forge Neo.
- `model_converter/core/`: požadavky, plán převodu, politiky a výsledkové reporty.
- `model_converter/io/`: inspekce safetensors, shard indexy, omezené čtení a zápis.
- `model_converter/adapters/`: modelové rodiny, mapování klíčů a komponent.
- `model_converter/profiles/`: verzované politiky výběru a zachování vrstev.
- `model_converter/backends/`: adaptéry konkrétních Kitchen API a detekce schopností.
- `model_converter/optimization/`: volitelný learned rounding a Prodigy+.
- `model_converter/validation/`: strukturální a numerická kontrola exportu.
- `model_converter/worker/`: samostatný proces, průběh, zrušení a chybové stavy.
- `tests/`: malé fixture kontraktů, integrační testy a volitelné GPU/ComfyUI scénáře.

Tok zpracování: inspekce → rozpoznání adaptéru → plán → kvantizace/optimalizace
→ zápis → strukturální kontrola → atomické dokončení → report.
Ověření generováním běží v odděleném cílovém ComfyUI při testování podporované kombinace.

## Milník 1: uzavření kontraktů a kompatibility

- Vybrat první modelovou rodinu a získat původní vstup i reprezentativní testovací workflow.
- Zaznamenat konkrétní verze/commity Forge Neo, Kitchen a cílového ComfyUI.
- Ověřit API z Kitchen nainstalované Forge Neo; nejnovější upstream není automaticky dostupný.
- Porovnat export StarNodes s oficiálními nástroji a loaderem pro každý plánovaný formát.
- Provést audit převzatých částí, licencí a jejich vazeb na ComfyUI.
- Rozhodnout, zda bude daná cesta převzatá, upravená nebo nově napsaná, a evidovat původ.

Výstup: matice model × vstup × formát × loader/verze × hardware, první fixture
kontraktů a konkrétní akceptační scénáře. Dosud neověřené varianty mají tento stav výslovně.

## Milník 2: vstup, plán a paměťově omezené I/O

- Implementovat čtení hlaviček bez načtení celého modelu.
- Validovat indexy shardů, chybějící/duplicitní tensory a nejednoznačné složky.
- Detekovat už kvantizovaný vstup a nepodporované dtypes před převodem.
- Implementovat profil a modelový adaptér; samotný blacklist nenahrazuje přemapování klíčů.
- Vytvořit dry-run plán s důvody výběru, fallbacky a odhadem výstupních bajtů.
- Navrhnout a ověřit writer: standardní `safetensors.save_file` nad celým výstupním
  slovníkem sám nezajišťuje omezenou RAM. Použít dvouprůchodový nebo spoolovaný zápis
  s korektními offsety a kontrolou výsledku běžným safetensors readerem.
- Řešit dočasný výstup, kolize názvů a atomické dokončení na stejném svazku.

Ověření: indexy shardů, deterministický plán, metadata/prefixy, zachování vstupu,
přerušený zápis a velký syntetický model s měřením skutečné špičkové RAM.

## Milník 3: základní formáty a export pro ComfyUI

- Připojit Kitchen přes vlastní tenký adaptér, bez závislosti na ComfyUI runtime.
- Implementovat FP8 E4M3FN a INT8 včetně ConvRot.
- Zapojit W4A8/W6A8 do stejného milníku, pokud odpovídající API i loader projdou kontrolami.
- Pro jednotlivé vrstvy kontrolovat rozměry, velikosti skupin, padding a možnost rotace.
- Použít canonical uložitelný layout, ne optimalizovaný resident-only layout.
- Zapisovat packed weights, škály a `comfy_quant` podle cílového kontraktu.
- U W4A8/W6A8 explicitně použít symetrickou variantu slučitelnou s loaderem;
  neexportovat závislost na nepodporovaném `weight_correction`.
- Zachovat nevybrané vrstvy podle profilu a vykázat každý fallback.

Ověření: malé matice a hraniční rozměry, kvantizace/dekvantizace, konečnost hodnot,
korektní scale keys, UTF-8 JSON konfigurace a reálné načtení/generování v ComfyUI.
Výsledky porovnávat s odpovídající upstream referencí, ne požadovat bitovou shodu
mezi backendy s různou aritmetikou bez důvodu.

## Milník 4: pracovní proces a rozhraní Forge Neo

- Spouštět worker stejným Python prostředím jako hostitel, s jasným protokolem událostí.
- Oddělit blokující převod od Gradio UI; řešit průběh, logy, zrušení a úklid vlastních
  dočasných souborů podle budoucího zadání implementace.
- Zobrazit vstup, profil, dostupné formáty, zařízení, výstup a dry-run plán.
- Rozlišit plánované ponechání vrstvy, náhradní formát a chybu převodu.
- Nabídnout přísný režim: selhání požadovaného převodu nesmí skončit skrytým úspěchem.
- Před spuštěním odhalit chybějící API bez automatických změn prostředí hostitele.

Ověření: instalace jako extension v podporovaném Forge Neo, pracovní proces,
zrušení i chyba workeru, cesty se znaky Unicode/mezerami a zobrazení skutečného reportu.

## Milník 5: další formáty a smíšené profily

- Přidat NVFP4, MXFP8 a INT4 ConvRot podle stejného exportního a validačního kontraktu.
- Rozlišit schopnost soubor vytvořit, načíst a zrychleně počítat na daném hardware.
- Doplnit per-layer smíšené profily a další výslovně ověřené modelové adaptéry.
- CPU nebo jiný fallback nabídnout pouze pro skutečně vyzkoušené cesty.
- W8A16 případně realizovat jako volbu výpočtu nad uložením INT8.
- AWQ/SVDQuant řešit samostatně; nepřebírat neověřený formát jen podle názvu z UI StarNodes.

Ověření: každý přidaný formát musí projít strukturální kontrolou i cílovým loaderem;
každá kombinace označená jako ověřená také generováním.

## Milník 6: volitelná optimalizace Prodigy+

- Zvolit algoritmus a metriku zvlášť od optimalizátoru: SVD vážená chyba,
  rekonstrukce výstupu vrstvy nebo kalibrovaná cesta mají různé náklady a význam.
- Nejprve podporovat omezenou sadu formátů. Nepředpokládat jeden vhodný postup pro všechny.
- Definovat učitelné parametry a diferencovatelnou aproximaci/relaxaci zaokrouhlení.
- Použít správnou mřížku cíle: INT8 a FP8 nelze optimalizovat totožným celočíselným round.
- Připojit Prodigy+ přímo jako volitelnou závislost, s připnutou ověřenou verzí.
- Explicitně nastavit Schedule-Free režim. CTQ v prověřených větvích používá
  `use_schedulefree=False`; výchozí nastavení balíčku nepovažovat za ekvivalent.
- Vyhodnocovat a uchovat kandidáty až po finálním zaokrouhlení; podle zvolené metriky
  se vrátit k základnímu výsledku, pokud optimalizace nepomohla nebo je nestabilní.
- Evidovat čas, paměť, seed, parametry a případná kalibrační data.

Ověření: konečnost gradientů/výstupu, skutečná změna uložených vah, metrika finálního
checkpointu, srovnání s převodem bez optimalizace a vizuální regresní vzorky.
Optimalizace nesmí blokovat používání základního převodu.

## Milník 7: vydání a dokumentace

- Popsat ověřené verze, modely a formáty včetně odlišení strukturální a vizuální validace.
- Dodat návod instalace, známá omezení, příklady profilů a přehled paměťových nároků.
- Otestovat čistou instalaci do Forge Neo bez nepřímé instalace celého ComfyUI.
- Uzavřít licencování vlastního projektu s vlastníkem a zkontrolovat attribution převzatého kódu.
- Označit první verzi až po splnění kritérií; experimentální volby musí být jasně označené.

## Pořadí a hranice

Milníky 1–4 tvoří cestu k první použitelné verzi; milníky 5–6 rozšiřují formáty
nebo kvalitu a mohou být přeuspořádány podle výsledků ověřování. Dokončení milníku
znamená splnění jeho ověření, nikoliv pouze napsaný kód. Žádné benchmarkové výsledky
ani obecné sliby vyšší kvality nejsou tímto plánem předjímány.
