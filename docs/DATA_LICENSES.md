# Data, model and artifact licences

Checked on **2026-11-17**. This repository redistributes **no third-party datasets and no third-party
weights** (`data/` and `checkpoints/` are git-ignored). It redistributes derived metrics only
(`experiments/results/*.json`). Each upstream artifact keeps its own licence; the table says what you
must do to reuse it. Nothing here is legal advice.

How each row was checked (column "Check"):

- **V** = read at the source URL on the date above (page, API field or licence file).
- **S** = secondary source only (aggregator, registry, search snippet); confirm before relying on it.
- **U** = not verified in this audit. Do not repeat the claim until someone has read the source.

Quoted wording is exact only where it is inside quotation marks and marked V; otherwise it is a
paraphrase of a page that was summarised by a tool.

## 1. Clay artifacts (what this paper's authors release)

| Artifact | Licence at source | Source URL | Check | Notes |
|---|---|---|---|---|
| Clay model code | Apache-2.0 | https://github.com/Clay-foundation/model (`LICENSE`; GitHub API `license.spdx_id` = `Apache-2.0`) | V | No `NOTICE` file at commit `f14e698f`. `LICENSE-MODEL.md` (OpenRAIL-M) existed from `12cc466` (2023-12-07, PR #63) until `8a2d50f` (2025-05-19, PR #351), which deleted it. |
| Clay v1.5 weights | Apache-2.0 | https://huggingface.co/made-with-clay/Clay (card header `license: apache-2.0`; HF revision `70200ebc`, last modified 2026-04-05) | V | Card body: "Clay model **weights** on Hugging Face ... License: Apache". The card's `license_link` points to `.../main/LICENSE.md`, which returns HTTP 404 (the file is `LICENSE`). Fix upstream. DOI per HF tags: `10.57967/hf/1537`. |
| Clay documentation | CC BY 4.0 | same model card, line "documentation ... License: CC-BY" | V | |
| Clay embeddings, per the HF model card | ODC-BY | same model card, line "embeddings on Source Cooperative ... License: ODC-BY" | V | **Conflicts** with the three catalog pages below (CC BY 4.0). The card is stale or the catalogs changed. Reconcile upstream. |
| `clay/lgnd-embeddings` (Source Cooperative) | "Licensed under CC BY 4.0." | https://source.coop/clay/lgnd-embeddings | V | Visibility: public. Maintained by "lgnd.ai" per its description. Page created 2026-06-01, updated 2026-06-02. |
| `clay/lgnd-clay-v1-5-sentinel-2-l2a` | "Licensed under CC BY 4.0." | https://source.coop/clay/lgnd-clay-v1-5-sentinel-2-l2a | V | Visibility: **unlisted** (`robots: noindex, nofollow`); its JSON-LD says `isAccessibleForFree: false` (meaning unclear; test anonymous access before saying "public"). Also listed in the AWS Open Data registry as "LGND Clay v1.5 Sentinel-2", licence "Creative Commons Attribution 4.0 International License", citation string "Clay v1.5 Sentinel-2. Source Cooperative. https://source.coop/clay/lgnd-clay-v1-5-sentinel-2-l2a" (registry file `datasets/lgnd-clay-v1-5-sentinel2.yaml`, S). |
| `clay/clay-v1-5-naip-2` | "License: Clay CC-By" (link to CC BY 4.0) | https://source.coop/clay/clay-v1-5-naip-2 | V | Visibility: public; updated 2025-08-21. Registry file `datasets/clay-v1-5-naip-2.yaml` also says CC BY 4.0 (S). Embeddings are derived from NAIP (public domain, see 3). |
| `Clay-foundation/stacchip` | Apache-2.0 | https://github.com/Clay-foundation/stacchip (GitHub API) | V | |

**Consequence for the manuscript.** "Apache-2.0 code, weights and catalogs" is wrong. Correct form:
Apache-2.0 for code and weights, CC BY 4.0 for the three embedding catalogs (as stated on each catalog
page), CC BY 4.0 for documentation. Attribution for a catalog: credit Clay / LGND, link the catalog
page, and carry through the upstream imagery notices in section 3.

### What the former OpenRAIL-M licence really contained

Text read from `Clay-foundation/model`, file `LICENSE-MODEL.md`, at parent commit `0a0b337a` of
`8a2d50f` (a copy is kept in the audit evidence, not in this repository). Title: "CLAY OpenRAIL-M
LICENSE", derived from "AI PUBS OPEN RAIL-M LICENSE". Attachment A ("Use restrictions") has exactly
three groups: **1 Legal** (violating applicable law), **2 Harm and Discrimination** (a to g: exploiting
minors, false information to Harm others, personal identifiable information to Harm, defamation or
harassment, Harm based on social behaviour or characteristics, exploiting vulnerabilities of a group,
discrimination on protected characteristics), **3 Transparency** (undisclosed machine-generated content,
impersonating humans, fully automated decisions that bind an individual).

The words "military" and "extractive" do **not** occur in either version of the file
(`12cc466` and `8a2d50f^`; the only hit is "extracted" in the "Data" definition) nor in upstream issue
#349 ("License Change: Update trained Clay Model License from OpenRAIL-M to Apache 2.0").

Those two restrictions **do** exist in another licence the paper discusses: the **OlmoEarth Artifact
License** (section 2): "(a) Any military and defense-related applications and use cases ..." and
"(b) Any extractive activities, operations and use cases involving the removal of raw materials from
the earth ...". The sentence in the manuscript that puts "restricted military" and "extractive use" in
quotation marks as clauses Clay had must be re-sourced (cite the OlmoEarth licence, or drop the
quotation marks and say "behavioural-use clauses such as those in other model licences").

## 2. Benchmark datasets named in Table 2, the README and FACT_SOURCES

| Dataset | Licence and attribution | Source URL | Check |
|---|---|---|---|
| GEO-Bench 1.0 (aggregate) | HF dataset card metadata `cc-by-sa-4.0`; card body empty. Per-dataset licences are in the paper's Table 1 (next rows). Code: Apache-2.0. | https://huggingface.co/datasets/recursix/geo-bench-1.0 ; https://arxiv.org/abs/2306.03831 ; https://github.com/ServiceNow/geo-bench | V |
| m-eurosat | MIT (Helber et al.) | arXiv:2306.03831 Table 1; https://github.com/phelber/EuroSAT (GitHub API: MIT) | V |
| m-so2sat | CC-BY-4.0 (Zhu et al. 2019) | arXiv:2306.03831 Table 1 | V (paper) |
| m-brick-kiln | CC-BY-SA 4.0 (Lee et al. 2021). Share-alike applies if you redistribute the data. | arXiv:2306.03831 Table 1 | V (paper) |
| m-forestnet | CC-BY-4.0 (Irvin et al. 2020) | arXiv:2306.03831 Table 1 | V (paper) |
| m-pv4ger, m-pv4ger-seg | MIT (Mayer et al. 2022) | arXiv:2306.03831 Table 1 | V (paper) |
| m-chesapeake-landcover | CDLA-Permissive-1.0 (Robinson et al. 2019) | arXiv:2306.03831 Table 1 | V (paper) |
| m-cashew-plantation | CC-BY-4.0 | arXiv:2306.03831 Table 1 | V (paper) |
| m-bigearthnet | CDLA-Permissive-1.0 (Sumbul et al. 2021) | arXiv:2306.03831 Table 1 | V (paper) |
| Copernicus-Bench: EuroSAT-S1, Biomass-S3, LC100Cls-S3 | HF card table lists each as CC BY 4.0; card says "See individual datasets". Code (Copernicus-FM repo): Apache-2.0. Underlying Sentinel-1 and Sentinel-3 data carry the Copernicus notice (section 3). Upstream licence of the label sources for Biomass-S3 and LC100Cls-S3 is not named on the card. | https://huggingface.co/datasets/wangyi111/Copernicus-Bench ; https://github.com/wangyi111/Copernicus-FM | V for the card, U for the label sources |
| EuroSAT-SAR (basis of EuroSAT-S1) | HF tag `mit`; EuroSAT original MIT | https://huggingface.co/datasets/wangyi111/EuroSAT-SAR | V (tag only) |
| Sen1Floods11 | **No licence found.** The GitHub repository has no licence file (GitHub API 404 for licence). README asks users to cite Bonafilia, Tellman, Anderson, Issenberg (2020), CVPR Workshops. Underlying Sentinel-1/2 are Copernicus data. Ask the maintainers before saying "open". | https://github.com/cloudtostreet/Sen1Floods11 | V (absence) |
| CropHarvest | "Creative Commons Attribution Share Alike 4.0 International" (Zenodo record; GitHub repo also CC-BY-SA-4.0). Share-alike on redistribution. Cite DOI 10.5281/zenodo.7257688 and Tseng et al. 2021. | https://zenodo.org/records/7257688 ; https://github.com/nasaharvest/cropharvest | V (record summary and GitHub API) |
| UCMerced Land Use | No formal licence. Imagery is from the USGS National Map Urban Area Imagery collection, which the dataset page describes as public domain. Cite Yang and Newsam (ACM SIGSPATIAL 2010). The dataset host did not respond in this audit. | http://weegee.vision.ucmerced.edu/datasets/landuse.html | S |
| RESISC-45 (NWPU-RESISC45) | **Not verified.** Licence and redistribution terms for the underlying images were not found. Cite Cheng, Han, Lu (Proc. IEEE 2017). Do not redistribute chips. | (original paper) | U |
| OlmoEarth v1.3 evaluation suite (45 tasks) | Each task has its own licence; not audited. The paper only quotes published numbers and uses a subset. | https://github.com/allenai/olmoearth_pretrain | U |
| AlphaEarth Foundations (AEF) Satellite Embedding | CC-BY 4.0. Required attribution: "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind." Cite Brown et al. 2025, arXiv:2507.22291. Mirror: Source Cooperative `tge-labs/aef` (registry: CC-BY 4.0). | https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL ; registry file `aef-source.yaml` | V (EE catalog page), S (registry) |
| TESSERA embeddings | Model and embeddings card (HF `geotessera/TESSERA-V-1.1`): `license: cc0-1.0`. Library: MIT (`ucam-eo/geotessera`, LICENSE.md). Cite Feng et al., arXiv:2506.20380. A search summary claimed an additional "Acceptable Use Policy"; the raw card text contains no such phrase, so treat that as unconfirmed. The AWS registry entry's licence field was not seen. | https://huggingface.co/geotessera/TESSERA-V-1.1 ; https://github.com/ucam-eo/geotessera | V (card header, MIT), U (AUP) |
| GEO-Bench-2, TerraTorch, VHR-Bench | Referenced in the text as external suites whose published numbers are quoted; no data from them is used. TerraTorch code: Apache-2.0 (GitHub API). The others were not audited. | https://github.com/IBM/terratorch | V (TerraTorch), U (others) |

## 3. Source imagery and reference layers used for pretraining and probes

| Source | Licence and attribution | Source URL | Check |
|---|---|---|---|
| Sentinel-1 / -2 / -3 (Copernicus) | EU law grants free, full and open access incl. reproduction, distribution, adaptation. Notice when distributing: 'Copernicus Sentinel data [Year]'. If modified: 'Contains modified Copernicus Sentinel data [Year]'. No warranty. | https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice (Regulation (EU) 1159/2013 art. 8) | V |
| Landsat 8/9 (USGS) | US government work. USGS states "USGS-authored or produced data and information are considered to be in the U.S. Public Domain." Requested credit: "Credit: U.S. Geological Survey". The Landsat-specific policy page was not retrieved. | https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits | V (general USGS), U (Landsat page) |
| NAIP | AWS Open Data registry licence field: "Public Domain with Attribution". The USDA programme page timed out. | https://registry.opendata.aws/naip/ (file `datasets/naip.yaml`) | S |
| LINZ / `nz-imagery` | Registry lists CC-BY-4.0. LINZ guidance: LINZ does not hold copyright in the imagery; attribute the actual licensor, e.g. "Sourced from the LINZ Data Service and licensed by (licensor), for re-use under the Creative Commons Attribution 4.0 International licence." Per-survey licensors vary. | https://registry.opendata.aws/nz-imagery/ ; https://www.linz.govt.nz/products-services/data/licensing-and-using-data/attributing-linz-basemaps-data | S |
| MODIS (MOD09A1) | NASA Earthdata data-use guidance: data from NASA-led missions "licensed as Creative Commons Zero (CC0)"; cite datasets; do not imply NASA endorsement. AWS registry `modis.yaml`: "There are no restrictions on the use of data, unless expressly identified prior to or at the time of receipt." | https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance | V |
| ESA WorldCover 2020 / 2021 | CC BY 4.0, "free of charge, without restriction of use". Map credit: "© ESA WorldCover project [year] / Contains modified Copernicus Sentinel data ([year]) processed by ESA WorldCover consortium". Cite the Zenodo DOIs on the data-access page (registry cites 10.5281/zenodo.5571936). | https://esa-worldcover.org/en/data-access | V |
| USPVDB (US Large-Scale Solar PV Database) | USGS and LBNL data release, DOI 10.5066/P9IA3TUS. Labels differ: data-release page CC0 1.0; energy.usgs.gov "public domain". The ScienceBase page returned HTTP 403 here. Cite the DOI and Fujita et al. 2023. | https://doi.org/10.5066/P9IA3TUS | S |
| NLCD (Annual NLCD Collection 1.x) | USGS EROS pages state no restrictions on use of the science products and ask for the citation "U.S. Geological Survey (USGS), 2024, Annual NLCD Collection 1 Science Products", DOI 10.5066/P94UXNTS. The specific NLCD vintage used by `run_clay_v15_experiments.py` must be recorded (see `CLAY_S2_NLCD_ROOT` builder) and cited. A rights statement saying "public domain" was not found. | https://www.usgs.gov/centers/eros/science/usgs-eros-archive-land-cover-annual-nlcd-collection-1-land-cover | S |
| Microsoft Planetary Computer (STAC access to S1 RTC, MODIS, NAIP index) | Terms of use not audited. Each collection carries its own licence field in the STAC item. | https://planetarycomputer.microsoft.com/ | U |

## 4. Model weights and code used as comparators or teachers

| Artifact | Licence | Source URL | Check | Obligation |
|---|---|---|---|---|
| DOFA (weights via TorchGeo) | HF cards `torchgeo/dofa` and `earthflow/DOFA`: CC-BY-4.0. Code: MIT. | https://huggingface.co/torchgeo/dofa ; https://github.com/zhu-xlab/DOFA | V | Attribute. |
| Copernicus-FM | HF card CC-BY-4.0; code Apache-2.0 | https://huggingface.co/wangyi111/Copernicus-FM | V | Attribute. |
| Panopticon | Code Apache-2.0 (`panopticon-FM/panopticon`); HF `lewaldm/panopticon` card says `mit`. Both permissive; note the mismatch. | https://github.com/panopticon-FM/panopticon | V | Keep notices. |
| Prithvi-EO-2.0 | HF cards (300M, 300M-TL): apache-2.0; repo `NASA-IMPACT/Prithvi-EO-2.0`: MIT | https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M | V | Keep notices. |
| **OlmoEarth** (v1, v1.2, v1.3 code, weights, data) | **OlmoEarth Artifact License (Ai2)**, not an OSI licence. Section 2 forbids "(a) Any military and defense-related applications ..." and "(b) Any extractive activities ...". Section 3: cite Ai2; link or include the agreement when distributing Artifacts; derivatives must carry the section 2 restrictions downstream. HF card of v1.2 Base: `license: other`, "licensed under the OlmoEarth Artifact License". | https://github.com/allenai/olmoearth_pretrain (LICENSE) ; https://huggingface.co/allenai/OlmoEarth-v1_2-Base | V | Do not publish OlmoEarth embeddings, outputs or fine-tuned weights from this repository without the section 2 restrictions. Benchmarking use and quoting scores is permitted by section 1. |
| **Scale-MAE** (code; FMOW_RGB weights used via TorchGeo) | **CC BY-NC 4.0** (repo LICENSE; HF mirror `isaaccorley/vit_large_patch16_224_fmow_rgb_scalemae`: cc-by-nc-4.0) | https://github.com/bair-climate-initiative/scale-mae | V | Non-commercial. Research comparison is fine; do not ship its weights or features in a commercial product; do not redistribute. |
| DINOv2 ViT-L/14-reg4 (Clay Stage-2 teacher; comparator) | Apache-2.0 (repo; HF `timm/vit_large_patch14_reg4_dinov2.lvd142m` and `facebook/dinov2-large`) | https://github.com/facebookresearch/dinov2 | V | Keep notices. |
| DINOv3 | **DINOv3 License (Meta)**, custom, gated access. HF cards: `license: other`, `dinov3-license`. | https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md | V | Check whether the paper ran the model or only quotes published numbers; only the latter needs no action. |
| Segment Anything (Stage-1 teacher, ViT-B/16) | Apache-2.0 | https://github.com/facebookresearch/segment-anything | V | Keep notices. |
| Galileo, TerraMind, CROMA, AnySat | Only published numbers are quoted. Licences not audited. | | U | |
| TorchGeo | MIT | https://github.com/microsoft/torchgeo | V | Keep notices. |
| timm | Apache-2.0 | https://github.com/huggingface/pytorch-image-models | U (known, not re-read) | |
| vit-pytorch, DOFA (vendored source in `claymodel/`) | MIT, "Copyright (c) 2020 Phil Wang"; MIT, "Copyright (c) 2024 Zhitong Xiong" | https://github.com/lucidrains/vit-pytorch ; https://github.com/zhu-xlab/DOFA | V | Reproduced in `NOTICE`. |

## 5. Figure 1 image assets

**Status (authors, 2026-11-17):** the four thumbnails are derived from Copernicus Sentinel imagery; the scene identifier and date were not recorded. Attribution: "Contains modified Copernicus Sentinel data". The notes below record how this was established and are kept for the record.

`figures/asset_chip_in.png`, `asset_chip_rec.png` (128 x 128), `asset_strip_full.png` (32 x 242),
`asset_strip_vis.png` (32 x 102). `generate_figures.py` only copies and places them; it does not create
them. They enter the history in the first commit (`31bd96c`) with no generating script, no source note
and no embedded metadata (PNG has no text chunks). Visual inspection of `asset_chip_in.png` shows a
false-colour coastal scene (near-infrared in red) consistent with a Sentinel-2 or Landsat chip, but the
sensor, scene and date are **unknown**. The original experiment scratch on the compute host contains no
file with those names.

Action required from the authors: identify the scene (re-create it with `ClayRuntime` from a recorded
STAC item) and then credit it as in section 3, or replace the assets with a scene whose source is
recorded. Until then the CC BY 4.0 grant in `NOTICE` excludes them.

Update: `figures/make_assets.py` (added by another work stream) regenerates `asset_chip_in.png` and
`asset_chip_rec.png` from one recorded NAIP item on the Planetary Computer and writes
`assets_provenance.json`; it does not overwrite the committed files. The two `asset_strip_*.png` files
are not covered by it. If the regenerated pair replaces the committed pair, credit NAIP as in section 3
("Public Domain with Attribution" per the AWS registry, S) and record the item id. The strips still need
a source.
