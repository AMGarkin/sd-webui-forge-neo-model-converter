# Pokyny pro práci na projektu

## Kontext a zdroj pravdy

Před implementací přečti `docs/PROJECT_BRIEF.md`, `docs/IMPLEMENTATION_PLAN.md`
a `docs/RESEARCH.md`. Dokumenty jsou česky; identifikátory kódu mohou být anglicky.
Změny rozsahu a důležitá rozhodnutí zaznamenej také do dokumentace.

Úvodní založení repozitáře je dokumentační. Implementaci zahaj až podle dalšího
zadání uživatele. Původní projekt `C:\Working\sd-webui-convert-to-quant`
není základem této implementace; nemaž jej a nekopíruj z něj automaticky kód.

## Architektonické zásady

- Jádro musí fungovat bez běžícího Forge Neo i ComfyUI. Integrace hostitele
  patří do samostatné vrstvy; do jádra neimportuj `folder_paths` ani `comfy.*`.
- Hlavní kvantizační závislost je `comfy-kitchen`; CTQ používej jako referenci,
  nikoliv automaticky jako runtime závislost.
- Prodigy+ je volitelný optimalizátor. Není náhradou algoritmu kvantizace,
  ztrátové funkce, kalibrace nebo exportu.
- Dostupnost formátů ověřuj podle konkrétních API a cílového loaderu.
  Nerozhoduj jen podle přítomnosti balíčku nebo názvu formátu.
- Respektuj závislosti hostitele. Automaticky neupgraduj Torch/CUDA/Kitchen
  v prostředí Forge Neo ani neinstaluj jeho konkurenční fork.
- Exportní kontrakt určuje ComfyUI: packed weights, škály, názvy tensorů
  a per-module `comfy_quant`. Přípona `.safetensors` kompatibilitu nedokazuje.
- Nepřepisuj zdrojové modely. Výstup zapisuj do dočasného souboru a zveřejni
  pod finálním názvem až po úspěšném dokončení a strukturální kontrole.
- Neponechávej selhanou konverzi skrytou za úspěšným stavem. Každý fallback
  musí být v plánu nebo závěrečném reportu viditelný.
- Navrhuj paměťově omezené zpracování a samostatný pracovní proces pro převod.

## Ověřování a převzatý kód

Testuj skutečné invarianty formátů, chybové stavy, paměťové chování a kompatibilitu
loaderu. Netvrď vizuální kvalitu ani rychlost bez odpovídajícího měření.
Rozlišuj dokončenou implementaci, strukturální validaci a ověření generováním.

Při převzetí upstream kódu eviduj zdroj, commit a zachovej příslušná licenční
upozornění. Profily a algoritmy ze StarNodes nebo CTQ nejsou automaticky důkazem
kompatibility všech jimi nabízených variant. Nezaváděj libovolnou vlastní licenci
projektu bez rozhodnutí vlastníka.
