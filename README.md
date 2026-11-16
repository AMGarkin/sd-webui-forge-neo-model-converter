# Forge Neo Model Converter

Convert and quantize diffusion models for ComfyUI.

Doplněk pro **SD WebUI Forge Neo** pro převod původních modelových vah
BF16 / FP16 / FP32 do kvantizovaných checkpointů `.safetensors` pro ComfyUI.

- Název projektu: **Forge Neo Model Converter**
- Repozitář a adresář doplňku: `sd-webui-forge-neo-model-converter`
- Název záložky v rozhraní: **Model Converter**
- GitHub: https://github.com/AMGarkin/sd-webui-forge-neo-model-converter

## Stav

**Příprava projektu, 16. 11. 2026.** Repozitář zatím obsahuje pouze zadání,
architektonická rozhodnutí a plán. Aplikace, instalační skripty ani runtime
závislosti dosud nejsou implementované.

## Dokumentace

- [Zadání a akceptační kritéria](docs/PROJECT_BRIEF.md)
- [Architektura a plán implementace](docs/IMPLEMENTATION_PLAN.md)
- [Zdroje, závěry průzkumu a otevřené otázky](docs/RESEARCH.md)
- [Pokyny pro další práci v repozitáři](AGENTS.md)

## Technický směr

Samostatné převodní jádro nad `comfy-kitchen`, tenká integrace do Forge Neo,
export podle specifikace ComfyUI a volitelná optimalizace s Prodigy+.
Vybrané části StarNodes Model Converter poslouží jako základ nebo reference
po kontrole správnosti, kompatibility a licenčních požadavků.
`convert_to_quant` není plánovanou povinnou závislostí.

Cílem není automaticky podporovat každý soubor z Hugging Face. Podpora bude
ověřovaná pro konkrétní kombinace modelové rodiny, struktury vstupu, formátu
výstupu a verze cílového ComfyUI.
