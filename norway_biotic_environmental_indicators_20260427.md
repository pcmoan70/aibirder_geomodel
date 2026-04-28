# Norway Biotic Environmental Indicator Features

**File:** `norway_biotic_environmental_indicators_20260427.md`  
**Purpose:** define indicator-species families, example taxa, and calculation methods for Norway-only biotic/ecological features in the Nordic + Norway residual geomodel.  
**Target model block:** `norway_biotic_indicator_*`

---

## 1. Scope

This document describes how to build biotic indicator features for a Norway-specialized bird occurrence model. The indicators are derived from non-bird observations, primarily:

- vascular plants
- bryophytes
- lichens
- fungi
- beetles and other saproxylic insects
- butterflies and moths
- dragonflies and aquatic insects
- molluscs
- selected other invertebrates

The indicators should not be raw "species X was observed here" variables. They should be smoothed, bias-controlled, ecological evidence surfaces.

The intended use is:

```text
shared Nordic model
    + Norway residual branch
        + topography
        + hydrology
        + climate
        + land cover
        + geology/minerals
        + non-bird biotic indicators
```

---

## 2. Important data-leakage rule

For a bird model:

```text
do not use bird observations as input features
```

Use only non-bird observations for these biotic features.

Recommended exclusions:

```text
taxon_group != Aves
```

Also build all indicator surfaces in a cross-fitted way:

```text
for validation fold k:
    build biotic surfaces without using non-bird observations from fold k
```

If the model is evaluated historically, use time-safe features:

```text
prediction year Y:
    only use non-bird observations observed before Y
```

---

## 3. General feature construction

For each H3 cell or raster/grid point `x`, radius `r`, and indicator group `G`:

```text
raw_score_G,r(x) =
    sum over observations o in group G:
        q_species(o)
      * q_record(o)
      * K_distance(dist(x, o), r)
      * K_time(age(o), tau_G)
```

where:

```text
K_distance(d, r) = exp(-d^2 / (2 r^2))
K_time(age, tau) = exp(-age / tau)
```

Then correct for observation effort:

```text
indicator_G,r(x) =
    log1p(raw_score_G,r(x))
  - beta_G * log1p(nonbird_effort_r(x))
```

or use a ratio:

```text
indicator_G,r(x) =
    log1p(weighted_indicator_count_G,r)
  - log1p(total_relevant_taxon_count_r)
```

Recommended radii:

```text
250 m      local habitat
1000 m     local landscape
5000 m     broader ecological context
```

For predators or large-ranging bird guilds, later add:

```text
10 km
25 km
```

but this document focuses on environmental indicators.

---

## 4. Observation-effort features

Always include effort controls alongside indicator scores:

```text
nonbird_observation_count_250m
nonbird_observation_count_1000m
nonbird_observation_count_5000m
nonbird_observer_count_1000m
nonbird_species_count_1000m
nonbird_observation_days_1000m
vascular_plant_effort_1000m
fungi_effort_1000m
lichen_effort_1000m
insect_effort_1000m
mollusc_effort_1000m
```

These help the model distinguish:

```text
high ecological quality
```

from:

```text
popular naturalist locality
```

---

## 5. Species weighting

Each species or family receives an indicator strength:

```text
q_species ∈ {0.25, 0.5, 1.0, 2.0}
```

Suggested interpretation:

| Weight | Meaning |
|---:|---|
| 0.25 | weak/generalist indicator |
| 0.5 | useful but broad indicator |
| 1.0 | good habitat indicator |
| 2.0 | strong specialist indicator |

For rare species, cap the maximum influence:

```text
q_species_effective = min(q_species, q_cap)
```

to avoid one rare observation dominating a large surface.

---

## 6. Indicator catalogue

The lists below are starting points, not final taxonomic truth. They should be converted into curated taxon lists using Norwegian names, scientific names, Artsdatabanken taxon IDs, GBIF taxon keys, and expert review.

For each indicator:

- **Positive evidence:** taxa that increase the score.
- **Negative or contrast evidence:** taxa that reduce or help separate the habitat.
- **Best taxon groups:** groups that are most informative.
- **Calculation:** recommended formula.
- **Caveats:** known pitfalls.

---

# 6.1 old_growth_forest_indicator

## Ecological meaning

Forest with long continuity, old trees, stable humid microclimate, structural complexity, rough bark, old stems, and slow turnover.

Norwegian forest biodiversity is strongly associated with forest age and old-tree structures. Old trees host richer communities of lichens, fungi, mosses and plants, and forests are the largest habitat category for red-listed species in Norway; many threatened forest species are specialists on dead wood, old trees, rich broad-leaved forest, or fire-created substrates.

## Best taxon groups

```text
lichens
wood-inhabiting fungi
bryophytes
saproxylic beetles
mycorrhizal fungi of old forest
```

## Useful families / groups

### Lichens

```text
Lobariaceae
Pannariaceae
Collemataceae / gelatinous cyanolichens
Arthoniaceae
Caliciaceae
Parmeliaceae, especially old-tree specialists
```

Examples / candidate genera:

```text
Lobaria
Peltigera
Nephroma
Leptogium
Pannaria
Ramalina
Usnea
Bryoria
Alectoria
Chaenotheca
Calicium
Arthonia
Thelotrema
```

### Fungi

```text
Polyporales
Hymenochaetales
Russulales wood-decay fungi
Thelephorales / old conifer forest fungi
Cortinariaceae in old rich forest
```

Candidate genera:

```text
Fomitopsis
Phellinus
Fomes
Antrodia
Skeletocutis
Amylocystis
Asterodon
Hericium
Phellodon
Hydnellum
Sarcodon
Cortinarius
```

### Beetles

```text
saproxylic beetles
longhorn beetles: Cerambycidae
bark beetle/predator communities: Curculionidae/Scolytinae
click beetles: Elateridae
darkling beetles: Tenebrionidae
false darkling beetles: Melandryidae
fungus beetles: Erotylidae, Ciidae
```

## Calculation

Use several sub-scores:

```text
old_tree_llichen_score
old_forest_fungi_score
old_forest_bryophyte_score
saproxylic_insect_score
```

Then:

```text
old_growth_forest_indicator =
    0.30 * old_tree_lichen_score
  + 0.30 * old_forest_fungi_score
  + 0.20 * saproxylic_insect_score
  + 0.20 * old_forest_bryophyte_score
```

Recommended radii:

```text
250 m
1000 m
5000 m
```

Use long time decay:

```text
tau = 25 to 50 years
```

because old-growth signals are persistent.

## Negative / contrast features

```text
clearcut_indicator
young_plantation_indicator
urban_effort_indicator
recent_forestry_disturbance
```

## Caveats

This indicator is highly affected by survey bias because rare lichens and fungi are searched for by specialists. Always include taxon-group effort controls.

---

# 6.2 deadwood_forest_indicator

## Ecological meaning

Forest with abundant dead wood, fallen logs, standing dead trees, decay stages, wood-decay fungi, and saproxylic insects.

## Best taxon groups

```text
wood-decay fungi
saproxylic beetles
deadwood mosses
deadwood lichens
```

## Useful families / groups

### Fungi

```text
Polyporaceae
Fomitopsidaceae
Hymenochaetaceae
Meruliaceae
Stereaceae
Phanerochaetaceae
Xylariaceae
```

Candidate genera:

```text
Fomitopsis
Phellinus
Fomes
Antrodia
Trametes
Ganoderma
Stereum
Xylaria
Kretzschmaria
Hericium
```

### Beetles

```text
Cerambycidae
Buprestidae
Elateridae
Tenebrionidae
Melandryidae
Tetratomidae
Ciidae
Erotylidae
Lucanidae
Staphylinidae linked to deadwood fungi
```

### Bryophytes / lichens

```text
Nowellia
Anastrophyllum
Calicium
Chaenotheca
Cladonia on decaying wood
```

## Calculation

```text
deadwood_forest_indicator =
    0.45 * deadwood_fungi_score
  + 0.35 * saproxylic_beetle_score
  + 0.10 * deadwood_bryophyte_score
  + 0.10 * deadwood_lichen_score
```

Use stronger local weighting than old-growth:

```text
250 m and 1000 m most important
5000 m useful as landscape continuity
```

Time decay:

```text
tau = 15 to 30 years
```

## Caveats

Some wood-decay fungi are common on managed forest stumps and do not indicate high-value deadwood habitat. Separate:

```text
general_deadwood_score
high_quality_deadwood_score
```

---

# 6.3 rich_deciduous_forest_indicator

## Ecological meaning

Warm, productive broad-leaved forest, often with elm, ash, lime, maple, oak, hazel, alder, rich field layer, rich fungi, molluscs, and insects.

## Best taxon groups

```text
vascular plants
molluscs
mycorrhizal fungi
lichens on broad-leaved trees
beetles associated with hollow/old deciduous trees
```

## Useful plant families / groups

```text
Ranunculaceae
Orchidaceae
Lamiaceae
Apiaceae
Primulaceae
Violaceae
Asparagaceae
Rosaceae
```

Candidate genera / species groups:

```text
Hepatica
Anemone
Actaea
Mercurialis
Sanicula
Primula
Viola
Lathyrus
Allium
Carex sylvatica group
orchids of rich forest
```

## Tree/habitat indicators

```text
Tilia cordata
Ulmus glabra
Fraxinus excelsior
Acer platanoides
Corylus avellana
Quercus robur
Alnus incana / Alnus glutinosa
```

## Molluscs

```text
Clausiliidae
Helicidae
Discidae
Vertiginidae
```

## Calculation

```text
rich_deciduous_forest_indicator =
    0.35 * rich_forest_flora_score
  + 0.20 * broadleaf_tree_indicator_score
  + 0.20 * mollusc_base_rich_forest_score
  + 0.15 * deciduous_tree_lichen_score
  + 0.10 * rich_forest_fungi_score
```

Recommended radii:

```text
250 m
1000 m
```

Time decay:

```text
tau = 20 to 40 years for flora/trees
tau = 10 to 25 years for fungi/insects
```

## Caveats

Rich deciduous forest and calcareous forest overlap. Keep both indicators but allow high correlation.

---

# 6.4 calcareous_forest_indicator

## Ecological meaning

Forest on lime-rich or base-rich substrate; often with calcium-demanding vascular plants, rich mycorrhizal fungi, orchids, molluscs, and distinctive forest vegetation.

## Best taxon groups

```text
vascular plants
orchids
mycorrhizal fungi
molluscs
calcium-demanding bryophytes
```

## Useful families / groups

### Plants

```text
Orchidaceae
Cyperaceae
Ranunculaceae
Gentianaceae
Primulaceae
```

Candidate genera/species groups:

```text
Cypripedium
Epipactis
Cephalanthera
Neottia
Listera
Hepatica
Actaea
Primula
Carex flacca / Carex rich-soil groups
```

### Fungi

```text
Cortinariaceae
Inocybaceae
Tricholomataceae
Hydnaceae / Bankeraceae
```

Candidate genera:

```text
Cortinarius
Inocybe
Tricholoma
Hydnellum
Sarcodon
Phellodon
Ramaria
```

### Molluscs

```text
Clausiliidae
Helicidae
Vertiginidae
```

## Calculation

```text
calcareous_forest_indicator =
    0.30 * calcareous_forest_flora_score
  + 0.25 * calcareous_mycorrhizal_fungi_score
  + 0.20 * mollusc_calcium_score
  + 0.15 * carbonate_bedrock_fraction
  + 0.10 * calcium_ca_p90_1000m
```

Recommended radii:

```text
250 m
1000 m
5000 m
```

## Caveats

This should also use abiotic geology/mineral features, not only species observations.

---

# 6.5 rich_fen_indicator

## Ecological meaning

Base-rich, often calcareous fen with high pH, brown mosses, calcium-demanding sedges, orchids, rich fen vascular plants, and sometimes molluscs. Semi-natural fens and lime-rich wetlands are conservation-relevant in Norway.

## Best taxon groups

```text
brown mosses
sedges
orchids
rich-fen vascular plants
molluscs
dragonflies in some cases
```

## Useful families / groups

### Bryophytes

```text
Amblystegiaceae
Calliergonaceae
Meesiaceae
Scorpidium group
Drepanocladus group
Campylium group
Tomentypnum group
Palustriella group
```

Candidate genera:

```text
Scorpidium
Drepanocladus
Campylium
Calliergon
Calliergonella
Tomentypnum
Palustriella
Meesia
Hamatocaulis
```

### Vascular plants

```text
Cyperaceae
Orchidaceae
Primulaceae
Gentianaceae
Parnassiaceae / Parnassia
```

Candidate genera/species groups:

```text
Carex rich-fen species
Schoenus
Eleocharis
Eriophorum latifolium group
Dactylorhiza
Epipactis
Gymnadenia
Parnassia
Primula farinosa
Triglochin
Tofieldia
```

### Molluscs

```text
Vertiginidae
Succineidae
small fen snails
```

## Calculation

```text
rich_fen_indicator =
    0.35 * rich_fen_bryophyte_score
  + 0.30 * rich_fen_vascular_plant_score
  + 0.15 * fen_orchid_score
  + 0.10 * fen_mollusc_score
  + 0.10 * calcium_or_base_richness_score
```

Recommended radii:

```text
250 m
1000 m
```

Time decay:

```text
tau = 15 to 40 years
```

## Caveats

Rich fens are often botanically well surveyed. Include bryophyte and vascular-plant effort controls.

---

# 6.6 poor_bog_indicator

## Ecological meaning

Acidic, nutrient-poor bog or poor fen dominated by Sphagnum, ericaceous shrubs, cotton-grasses, and low base-richness.

## Best taxon groups

```text
Sphagnum mosses
ericaceous plants
bog sedges
peatland specialist insects
```

## Useful families / groups

### Bryophytes

```text
Sphagnaceae
```

Candidate taxa:

```text
Sphagnum fuscum
Sphagnum magellanicum aggregate
Sphagnum rubellum
Sphagnum capillifolium
Sphagnum papillosum
Sphagnum tenellum
Sphagnum balticum
```

### Vascular plants

```text
Ericaceae
Cyperaceae
Droseraceae
Scheuchzeriaceae
```

Candidate genera/species groups:

```text
Calluna
Erica
Vaccinium
Andromeda
Oxycoccus
Empetrum
Eriophorum vaginatum
Eriophorum angustifolium
Trichophorum
Drosera
Scheuchzeria
Rhynchospora
```

### Insects

```text
Odonata associated with acidic bog pools
specialist Lepidoptera of bog/heath plants
```

## Calculation

```text
poor_bog_indicator =
    0.40 * sphagnum_bog_score
  + 0.25 * ericaceous_bog_flora_score
  + 0.20 * bog_sedge_score
  + 0.10 * acidic_wetland_insect_score
  - 0.15 * rich_fen_indicator
```

Recommended radii:

```text
250 m
1000 m
5000 m
```

## Caveats

Poor bog and wet heath may overlap. Use hydrology and peatland maps where available.

---

# 6.7 marshland_reedbed_indicator

## Ecological meaning

Emergent wetland vegetation: reedbeds, tall sedges, cattails, swamp, shallow water margins, eutrophic marsh.

## Best taxon groups

```text
vascular plants
aquatic beetles
dragonflies
molluscs
dipterans if available
```

## Useful families / groups

### Vascular plants

```text
Poaceae
Cyperaceae
Typhaceae
Juncaceae
Alismataceae
```

Candidate genera/species groups:

```text
Phragmites
Typha
Schoenoplectus
Bolboschoenus
Carex acuta / rostrata / vesicaria groups
Glyceria
Phalaris
Iris pseudacorus
Alisma
Sparganium
```

### Insects

```text
Odonata
Dytiscidae
Hydrophilidae
Chironomidae, if reliable
```

### Molluscs

```text
Lymnaeidae
Planorbidae
Bithyniidae
Valvatidae
```

## Calculation

```text
marshland_reedbed_indicator =
    0.45 * reedbed_vascular_plant_score
  + 0.20 * tall_sedge_swamp_score
  + 0.15 * aquatic_insect_score
  + 0.10 * freshwater_mollusc_score
  + 0.10 * wetland_map_fraction
```

Recommended radii:

```text
250 m
1000 m
```

Time decay:

```text
tau = 10 to 20 years
```

## Caveats

Reedbeds can change quickly with water level, eutrophication, grazing, and succession. Use shorter time decay than old forest.

---

# 6.8 wet_meadow_indicator

## Ecological meaning

Semi-natural or natural wet meadow, seasonally wet grassland, hay meadow, grazed wet pasture, often sedges/rushes/forbs and high insect productivity.

## Best taxon groups

```text
vascular plants
butterflies/moths
bees/hoverflies if available
wet meadow molluscs
```

## Useful families / groups

### Plants

```text
Cyperaceae
Juncaceae
Poaceae
Ranunculaceae
Fabaceae
Plantaginaceae
Orchidaceae
Asteraceae
```

Candidate genera/species groups:

```text
Juncus
Carex
Caltha
Ranunculus
Filipendula
Succisa
Lychnis
Dactylorhiza
Valeriana
Geum rivale
Trifolium
Lotus
```

### Insects

```text
Lepidoptera meadow specialists
Syrphidae
Apoidea
```

## Calculation

```text
wet_meadow_indicator =
    0.40 * wet_meadow_flora_score
  + 0.20 * semi_natural_management_flora_score
  + 0.20 * pollinator_meadow_score
  + 0.10 * wetness_topographic_score
  + 0.10 * grassland_map_fraction
```

Recommended radii:

```text
250 m
1000 m
```

## Caveats

Distinguish species-rich wet meadow from fertilized wet pasture. Add negative evidence:

```text
high_nutrient_ruderal_score
intensive_agriculture_fraction
```

---

# 6.9 eutrophic_freshwater_indicator

## Ecological meaning

Nutrient-rich lakes, ponds, river margins, and aquatic vegetation with high productivity.

## Best taxon groups

```text
aquatic vascular plants
molluscs
dragonflies
aquatic beetles
algae/macrophyte indicators if available
```

## Useful families / groups

### Aquatic plants

```text
Potamogetonaceae
Hydrocharitaceae
Nymphaeaceae
Lemnaceae / Araceae
Ceratophyllaceae
Alismataceae
```

Candidate genera/species groups:

```text
Potamogeton broad-leaved species
Stuckenia
Myriophyllum
Ceratophyllum
Lemna
Spirodela
Nuphar
Nymphaea
Elodea
Sagittaria
Sparganium
```

### Molluscs

```text
Lymnaeidae
Planorbidae
Bithyniidae
Valvatidae
Unionidae where relevant
```

### Insects

```text
Odonata
Dytiscidae
Hydrophilidae
Corixidae / aquatic Hemiptera
```

## Calculation

```text
eutrophic_freshwater_indicator =
    0.35 * eutrophic_macrophyte_score
  + 0.20 * freshwater_mollusc_score
  + 0.20 * aquatic_insect_score
  + 0.15 * reedbed_marsh_score
  + 0.10 * nutrient_runoff_or_agriculture_proxy
```

Recommended radii:

```text
250 m
1000 m
5000 m
```

## Caveats

High eutrophic score can mean natural productivity or pollution. Keep it as an ecological productivity feature, not necessarily a positive habitat-quality feature.

---

# 6.10 oligotrophic_freshwater_indicator

## Ecological meaning

Nutrient-poor lakes, ponds, clear waters, acidic or soft-water systems, low productivity, often with isoetids and sparse aquatic vegetation.

## Best taxon groups

```text
aquatic vascular plants
bryophytes
dragonflies
acidic-water insects
```

## Useful families / groups

### Aquatic plants

```text
Isoetaceae
Plantaginaceae
Littorellaceae concept
Juncaginaceae
Potamogetonaceae, narrow soft-water species
```

Candidate genera/species groups:

```text
Isoetes
Lobelia dortmanna
Littorella
Subularia
Juncus bulbosus
Sparganium angustifolium group
Potamogeton polygonifolius group
```

### Bryophytes

```text
aquatic mosses in soft-water systems
Sphagnum around lake margins
```

### Insects

```text
Odonata of oligotrophic/acidic lakes
Trichoptera if available
Ephemeroptera/Plecoptera where reliable
```

## Calculation

```text
oligotrophic_freshwater_indicator =
    0.45 * soft_water_macrophyte_score
  + 0.20 * oligotrophic_bryophyte_score
  + 0.15 * acidic_water_insect_score
  + 0.10 * low_agriculture_buffer_score
  - 0.10 * eutrophic_freshwater_indicator
```

Recommended radii:

```text
250 m
1000 m
5000 m
```

## Caveats

Lake chemistry data, if available, is better than species-only inference. Use geology/mineral and catchment features.

---

# 6.11 coastal_heath_indicator

## Ecological meaning

Open coastal heathland shaped by long-term grazing and burning, often Calluna-dominated, oceanic, low tree cover, with heathland flora and invertebrates.

Coastal heath is listed among threatened semi-natural ecosystem types in Norway and is dependent on traditional management such as grazing and heather burning.

## Best taxon groups

```text
vascular plants
bryophytes
lichens
heathland butterflies/moths
bees/wasps where available
```

## Useful families / groups

### Plants

```text
Ericaceae
Cyperaceae
Poaceae
Fabaceae
Juncaceae
```

Candidate taxa:

```text
Calluna vulgaris
Erica tetralix
Empetrum nigrum
Vaccinium spp.
Molinia caerulea
Nardus stricta
Trichophorum
Carex panicea group
Genista / Ulex if regionally relevant
```

### Bryophytes / lichens

```text
Sphagnum in wet heath
Cladonia
Racomitrium
```

### Insects

```text
heathland Lepidoptera
wild bees linked to open heath
```

## Calculation

```text
coastal_heath_indicator =
    0.35 * heath_flora_score
  + 0.20 * open_low_tree_cover_score
  + 0.15 * oceanic_coastal_zone_score
  + 0.15 * heathland_invertebrate_score
  + 0.10 * burning/grazing_management_proxy
  - 0.10 * forest_regrowth_score
```

Recommended radii:

```text
250 m
1000 m
5000 m
```

## Caveats

Calluna alone is too broad. Combine with coastal/oceanic location, low tree cover, and management indicators.

---

# 6.12 saltmarsh_shore_meadow_indicator

## Ecological meaning

Saltmarsh, tidal meadow, semi-natural tidal meadow, upper littoral meadow; often saline-tolerant plants, grazing/haymaking history, shore meadow fauna.

Semi-natural tidal and salt meadow is an endangered semi-natural ecosystem type in Norway; tidal meadow is also assessed as vulnerable.

## Best taxon groups

```text
halophytic vascular plants
coastal meadow plants
ground beetles
shore flies if available
molluscs
```

## Useful families / groups

### Plants

```text
Amaranthaceae / Chenopodiaceae concept
Plumbaginaceae
Juncaceae
Poaceae
Cyperaceae
Asteraceae
Plantaginaceae
```

Candidate genera/species groups:

```text
Salicornia
Suaeda
Atriplex
Spergularia
Triglochin
Plantago maritima
Armeria
Juncus gerardii
Puccinellia
Festuca rubra saltmarsh forms
Glaux / Lysimachia maritima
Bolboschoenus
Carex paleacea group
```

### Invertebrates

```text
Carabidae of saltmarsh
Diptera shore groups if available
coastal molluscs in upper shore
```

## Calculation

```text
saltmarsh_shore_meadow_indicator =
    0.50 * halophyte_flora_score
  + 0.20 * tidal_zone_distance_score
  + 0.15 * shore_meadow_invertebrate_score
  + 0.10 * grazing_management_proxy
  - 0.10 * reedbed_or_forest_regrowth_score
```

Recommended radii:

```text
100 m
250 m
1000 m
```

## Caveats

Spatial precision matters. Use coastline, elevation, tidal zone, and land-cover masks to avoid mapping saltmarsh too far inland.

---

# 6.13 sand_dune_indicator

## Ecological meaning

Coastal dunes, sandy beaches, semi-stabilized dunes, dry sandy grassland, dune slacks.

## Best taxon groups

```text
vascular plants
ground beetles
solitary bees/wasps
spiders if available
moths/butterflies
```

## Useful families / groups

### Plants

```text
Poaceae
Caryophyllaceae
Asteraceae
Fabaceae
Brassicaceae
Plantaginaceae
```

Candidate genera/species groups:

```text
Ammophila
Leymus
Elymus
Honckenya
Cakile
Salsola/Kali
Festuca sandy grassland species
Carex arenaria
Eryngium if present regionally
Lotus
Trifolium arvense group
Thymus
```

### Invertebrates

```text
Carabidae
Tenebrionidae
Apoidea
Sphecidae/Crabronidae
Noctuidae dune species
```

## Calculation

```text
sand_dune_indicator =
    0.40 * dune_flora_score
  + 0.20 * sandy_substrate_score
  + 0.20 * dune_invertebrate_score
  + 0.10 * coastal_distance_score
  - 0.10 * closed_forest_or_urban_cover
```

Recommended radii:

```text
100 m
250 m
1000 m
```

## Caveats

Very sensitive to geolocation error. Use sand/substrate maps and coastal morphology.

---

# 6.14 alpine_heath_indicator

## Ecological meaning

Alpine dwarf-shrub heath, exposed alpine vegetation, wind-exposed ridges, low shrubs, graminoids, lichens and mosses.

## Best taxon groups

```text
vascular plants
lichens
bryophytes
butterflies/moths
```

## Useful families / groups

### Plants

```text
Ericaceae
Cyperaceae
Poaceae
Juncaceae
Salicaceae dwarf willows
```

Candidate taxa:

```text
Empetrum nigrum
Vaccinium uliginosum
Vaccinium vitis-idaea
Arctous alpinus
Loiseleuria procumbens
Cassiope hypnoides
Dryas octopetala, if calcareous
Betula nana
Salix herbacea / reticulata / polaris groups
Carex bigelowii
Juncus trifidus
```

### Lichens

```text
Cladonia
Cetraria
Flavocetraria
Alectoria
Stereocaulon
```

### Bryophytes

```text
Racomitrium
Polytrichum
Dicranum
```

## Calculation

```text
alpine_heath_indicator =
    0.35 * alpine_heath_flora_score
  + 0.20 * alpine_lichen_score
  + 0.15 * alpine_bryophyte_score
  + 0.15 * elevation_above_treeline_score
  + 0.10 * low_tree_cover_score
```

Recommended radii:

```text
250 m
1000 m
5000 m
```

## Caveats

Separate from alpine snowbed and calcareous alpine using snowbed species and calcium indicators.

---

# 6.15 alpine_snowbed_indicator

## Ecological meaning

Late snow-lie habitats, short growing season, snowbed bryophytes and vascular plants, wet snowbed/snowbed spring systems.

Snowbed is included among sparsely vegetated mountain ecosystem types in the Norwegian Red List ecosystem framework.

## Best taxon groups

```text
vascular plants
bryophytes
snowbed lichens
```

## Useful families / groups

### Plants

```text
Salicaceae dwarf willows
Cyperaceae
Juncaceae
Saxifragaceae
Orobanchaceae / alpine herbs
```

Candidate taxa/genera:

```text
Salix herbacea
Salix polaris
Gnaphalium / Omalotheca supina
Sibbaldia
Saxifraga
Carex lachenalii group
Juncus biglumis / triglumis groups
Ranunculus glacialis in extreme sites
```

### Bryophytes

```text
Anthelia
Kiaeria
Marsupella
Nardia
Polytrichum
Scapania
```

## Calculation

```text
alpine_snowbed_indicator =
    0.40 * snowbed_flora_score
  + 0.25 * snowbed_bryophyte_score
  + 0.20 * snow_persistence_proxy
  + 0.10 * concave_topography_score
  - 0.10 * dry_exposed_ridge_score
```

Recommended radii:

```text
100 m
250 m
1000 m
```

Time decay:

```text
tau = 15 to 30 years
```

## Caveats

Climate warming may shift snowbed vegetation. Use snow persistence from remote sensing where possible.

---

# 6.16 calcareous_alpine_indicator

## Ecological meaning

Alpine habitats on base-rich or calcareous substrate, often very species-rich, with Dryas, Saxifraga, rich sedges, calcicolous bryophytes, and molluscs.

## Best taxon groups

```text
vascular plants
bryophytes
lichens
molluscs
```

## Useful families / groups

### Plants

```text
Rosaceae
Saxifragaceae
Cyperaceae
Brassicaceae
Ranunculaceae
Orchidaceae in lower alpine/calcareous sites
```

Candidate taxa/genera:

```text
Dryas octopetala
Saxifraga oppositifolia
Saxifraga aizoides
Saxifraga paniculata
Carex rupestris group
Carex capillaris group
Kobresia / Carex nardina group where relevant
Silene acaulis
Bartsia alpina
Tofieldia
Pinguicula
```

### Bryophytes

```text
Tortella
Ctenidium
Scorpidium in wet calcareous sites
Drepanocladus rich-wet groups
```

### Molluscs

```text
Vertiginidae
Clausiliidae
small calcicolous land snails
```

## Calculation

```text
calcareous_alpine_indicator =
    0.35 * calcicolous_alpine_flora_score
  + 0.20 * calcareous_bryophyte_score
  + 0.15 * alpine_mollusc_calcium_score
  + 0.15 * carbonate_bedrock_fraction
  + 0.15 * calcium_ca_p90_1000m
```

Recommended radii:

```text
250 m
1000 m
5000 m
```

## Caveats

Should be strongly conditioned on altitude/bioclimatic zone. Dryas can also occur below alpine in special habitats.

---

# 6.17 dry_warm_slope_indicator

## Ecological meaning

South-facing, dry, warm, often base-rich slopes with thermophilous plants, dry grassland insects, solitary bees/wasps, butterflies, and sometimes open deciduous scrub.

## Best taxon groups

```text
vascular plants
butterflies
bees/wasps
grasshoppers
ground beetles
molluscs if calcareous
```

## Useful families / groups

### Plants

```text
Lamiaceae
Fabaceae
Caryophyllaceae
Asteraceae
Rosaceae
Poaceae
Orchidaceae
```

Candidate genera/species groups:

```text
Thymus
Origanum
Clinopodium
Lotus
Anthyllis
Trifolium dry grassland groups
Silene
Dianthus
Helianthemum
Fragaria
Artemisia
Pulsatilla
```

### Insects

```text
Lepidoptera dry-grassland species
Apoidea
Sphecidae/Crabronidae
Orthoptera
Carabidae
```

### Molluscs

```text
Helicidae
Clausiliidae
Pupillidae
```

## Calculation

```text
dry_warm_slope_indicator =
    0.30 * dry_slope_flora_score
  + 0.25 * thermophilous_insect_score
  + 0.15 * south_facing_slope_score
  + 0.10 * low_soil_moisture_score
  + 0.10 * calcareousness_index
  + 0.10 * open_land_fraction
```

Recommended radii:

```text
100 m
250 m
1000 m
```

## Caveats

In Norway, many dry warm slopes are rare and patchy. Use fine topography and solar exposure.

---

# 6.18 semi_natural_grassland_indicator

## Ecological meaning

Traditionally managed, species-rich meadow or pasture with long continuity, low fertilizer input, grazing or haymaking, and high plant/insect richness.

Semi-natural grassland and hayfields are highly threatened in Norway; hayfields have been assessed as critically endangered due to strong area reduction, and current monitoring shows many grasslands are abandoned and overgrown.

## Best taxon groups

```text
vascular plants
butterflies/moths
bees
hoverflies
grassland fungi
```

## Useful families / groups

### Plants

```text
Fabaceae
Orchidaceae
Asteraceae
Lamiaceae
Plantaginaceae
Caryophyllaceae
Gentianaceae
Poaceae, non-dominant species-rich grassland groups
```

Candidate genera/species groups:

```text
Anthyllis
Lotus
Trifolium
Vicia
Lathyrus
Leucanthemum
Campanula
Primula
Gentianella
Dactylorhiza
Gymnadenia
Platanthera
Pilosella
Thymus
Succisa
Rhinanthus
Plantago media
```

### Fungi

```text
waxcaps: Hygrophoraceae
Entolomataceae
Clavariaceae
Geoglossaceae
```

Candidate genera:

```text
Hygrocybe
Cuphophyllus
Gliophorus
Entoloma
Clavaria
Clavulinopsis
Geoglossum
Microglossum
```

### Insects

```text
Lepidoptera grassland specialists
Apoidea
Syrphidae
Orthoptera
```

## Calculation

```text
semi_natural_grassland_indicator =
    0.35 * semi_natural_grassland_flora_score
  + 0.20 * grassland_fungi_score
  + 0.20 * grassland_pollinator_score
  + 0.10 * low_nutrient_indicator_score
  + 0.10 * open_grassland_map_fraction
  - 0.10 * regrowth_shrub_tree_score
  - 0.10 * high_fertilization_ruderal_score
```

Recommended radii:

```text
100 m
250 m
1000 m
```

Time decay:

```text
tau = 10 to 25 years
```

## Caveats

Many semi-natural grasslands are small. Use high spatial precision, and distinguish them from fertilized improved grassland.

---

# 6.19 urban_ruderal_indicator

## Ecological meaning

Disturbed urban/suburban habitat, road verges, waste ground, railways, construction sites, nutrient-rich disturbed soils, alien species, ruderal flora.

## Best taxon groups

```text
vascular plants
alien/invasive plants
ruderal insects
synanthropic molluscs
```

## Useful families / groups

### Plants

```text
Asteraceae
Brassicaceae
Polygonaceae
Amaranthaceae
Fabaceae
Poaceae
Plantaginaceae
Urticaceae
```

Candidate genera/species groups:

```text
Taraxacum
Cirsium
Artemisia
Tripleurospermum
Senecio
Solidago
Erigeron
Bunias
Capsella
Sisymbrium
Chenopodium
Atriplex
Rumex
Polygonum
Urtica
Plantago major
Poa annua
Bromus
Lolium
Impatiens
Reynoutria / Fallopia
Heracleum alien species
Lupinus
```

### Insects / molluscs

```text
synanthropic butterflies/moths
urban bees
slugs/snails associated with gardens and disturbed habitats
```

## Calculation

```text
urban_ruderal_indicator =
    0.35 * ruderal_flora_score
  + 0.20 * alien_species_score
  + 0.15 * urban_landcover_fraction
  + 0.10 * road_rail_distance_score
  + 0.10 * nutrient_disturbance_score
  + 0.10 * urban_invertebrate_score
```

Recommended radii:

```text
100 m
250 m
1000 m
```

Time decay:

```text
tau = 3 to 10 years
```

## Caveats

Urban ruderal indicators may strongly correlate with observation effort. Effort correction is mandatory.

---

## 7. Calculation pipeline

### Step 1: Build curated taxon lists

For each indicator group:

```text
indicator_id
taxon_key
scientific_name
norwegian_name
taxon_rank
taxon_group
indicator_weight
positive_or_negative
notes
source
```

Example:

```csv
indicator_id,taxon_group,scientific_name,indicator_weight,sign
old_growth_forest_indicator,lichen,Lobaria pulmonaria,2.0,+1
deadwood_forest_indicator,fungus,Fomitopsis rosea,2.0,+1
rich_fen_indicator,bryophyte,Scorpidium scorpioides,2.0,+1
semi_natural_grassland_indicator,fungus,Hygrocybe spp.,1.0,+1
urban_ruderal_indicator,plant,Reynoutria japonica,1.0,+1
```

### Step 2: Clean occurrence data

Required filters:

```text
country == Norway
coordinate present
coordinate uncertainty <= threshold
date present or use weaker weight
taxon group != birds
remove obvious cultivated/garden-only records where necessary
remove or flag captive/cultivated records
deduplicate by taxon, observer, day, grid cell
```

Recommended coordinate uncertainty thresholds:

```text
fine indicators:
    <= 100 m or <= 250 m

broad indicators:
    <= 1000 m

rare old records:
    allow larger uncertainty but downweight
```

### Step 3: Create effort surfaces

By taxon group and radius:

```text
effort_vascular_plants_250m
effort_bryophytes_250m
effort_lichens_250m
effort_fungi_250m
effort_insects_250m
effort_molluscs_250m
```

### Step 4: Compute raw indicator surfaces

For each grid point/H3 cell:

```text
raw_indicator_G_r =
    Σ weighted observations within radius r
```

### Step 5: Effort correction

Preferred first version:

```text
indicator_G_r =
    log1p(raw_indicator_G_r)
  - beta_G * log1p(effort_relevant_taxa_r)
```

Estimate `beta_G` from validation or use:

```text
beta_G = 0.5
```

as a starting value.

### Step 6: Normalize

Use robust normalization:

```text
indicator_norm =
    clip((indicator - median) / IQR, -5, 5)
```

or quantile transform to `[0, 1]`.

### Step 7: Export features

Recommended columns:

```text
old_growth_forest_indicator_250m
old_growth_forest_indicator_1000m
old_growth_forest_indicator_5000m
deadwood_forest_indicator_250m
...
urban_ruderal_indicator_1000m
```

Also export uncertainty:

```text
old_growth_forest_indicator_effort_1000m
old_growth_forest_indicator_n_records_1000m
old_growth_forest_indicator_uncertainty_1000m
```

---

## 8. Indicator uncertainty

Define an uncertainty value:

```text
uncertainty_G_r =
    1 / sqrt(1 + effort_relevant_taxa_r)
```

or:

```text
uncertainty_G_r =
    exp(-effort_relevant_taxa_r / effort_scale_G)
```

Use this in the Norway residual gate:

```text
low uncertainty -> stronger Norway biotic residual
high uncertainty -> weaker Norway biotic residual
```

---

## 9. Use in Norway residual model

Add a biotic encoder:

```python
biotic_embedding = BioticIndicatorEncoder(
    indicator_scores,
    effort_features,
    uncertainty_features
)
```

Then:

```text
norway_residual_embedding =
    concat(
        topography_embedding,
        hydrology_embedding,
        ecosystem_embedding,
        geology_mineral_embedding,
        biotic_embedding
    )
```

Final Norway correction:

```text
z_final =
    z_shared_nordic
  + gate_NO * z_residual_NO
```

The gate should receive:

```text
biotic_uncertainty
feature_missing_masks
local_observation_density
species_sample_count
```

---

## 10. Biotic-aware smoothing

Update Norway residual smoothing:

```text
w_NO =
    w_geo
  * w_alt
  * w_ecosystem
  * w_geochem
  * w_biotic
```

where:

```text
w_biotic =
    exp(-||biotic_embedding_i - biotic_embedding_j||^2 / sigma_biotic^2)
```

This prevents smoothing between sites that are close but biologically different.

---

## 11. Recommended v1 indicator set

Use all 19 requested indicators, but prioritize these for the first production-quality run:

```text
old_growth_forest_indicator
deadwood_forest_indicator
rich_deciduous_forest_indicator
calcareous_forest_indicator
rich_fen_indicator
poor_bog_indicator
marshland_reedbed_indicator
semi_natural_grassland_indicator
coastal_heath_indicator
saltmarsh_shore_meadow_indicator
alpine_heath_indicator
alpine_snowbed_indicator
calcareous_alpine_indicator
dry_warm_slope_indicator
urban_ruderal_indicator
```

The freshwater and sand-dune indicators are useful but may need stronger map/hydrology/substrate constraints.

---

## 12. Validation

For each indicator, produce maps and diagnostic plots:

```text
indicator distribution
indicator vs effort
indicator vs known NiN / land-cover class
indicator vs altitude
indicator vs geology/calcium
indicator vs bird residual improvement
```

Use spatial holdout validation:

```text
hold out municipalities or H3 blocks
build indicator surfaces without holdout records
test whether indicator improves bird prediction in holdout
```

Do not accept an indicator simply because it improves random validation; random validation may reward observer-bias leakage.

---

## 13. Source notes

This plan is grounded in the following Norwegian/ecological source structure:

- Artsdatabanken provides Norwegian species information, species descriptions, identification material, and species-group pages, including lichens, fungi, plants and invertebrates.
- Artsdatabanken's 2021 Red List summary notes that forests contain the largest share of threatened species in Norway; the largest threatened forest groups include fungi, beetles, lichens, dipterans and lepidopterans, with many species specialized on dead wood, large broad-leaved trees, or burnt areas.
- Artsdatabanken's ecosystem red-list pages describe forest, wetlands, semi-natural landscapes, and sparsely vegetated habitats including snowbeds and sand dunes.
- The forest ecosystem description emphasizes forestry impacts on old trees, dead wood and substrate availability.
- The semi-natural landscapes description emphasizes that semi-natural grassland, hayfields, coastal heath and semi-natural tidal/salt meadow depend on long-term extensive management such as grazing, haymaking or heather burning.
- The wetlands description defines wetlands by high water table or stable surface water and includes semi-natural fens and other wetland ecosystem types.
- NIBIO has recent monitoring information on Norway's species-rich semi-natural grasslands and reports strong decline/encroachment.
- NIBIO research summaries highlight that old trees support diversity of lichens, fungi, mosses and plants.

URLs used while preparing this document:

```text
https://artsdatabanken.no/Pages/195122/
https://artsdatabanken.no/Pages/135386/
https://artsdatabanken.no/Pages/318297/
https://artsdatabanken.no/Pages/318299/
https://artsdatabanken.no/Pages/317603/
https://artsdatabanken.no/Pages/317602/Sparsely_vegetated_habitats
https://www.nibio.no/en/about-eng/research-matters/division-of-forest-and-forest-resources/research-matters-forest-and-forest-resources-2024/old-trees-boost-biodiversity
https://www.nibio.no/en/news/2026/norways-species-rich-grasslands-in-decline-new-monitoring-reveals-alarming-trends
```

---

## 14. Practical next step

Create a CSV seed file:

```text
norway_indicator_taxa_seed.csv
```

with columns:

```text
indicator_id
taxon_group
family
genus
species
scientific_name
norwegian_name
taxon_key
indicator_weight
sign
radius_default_m
time_decay_years
notes
source
```

Then iteratively improve it with:

```text
expert review
Artsdatabanken taxon matching
GBIF taxon matching
spatial validation
bird-model residual diagnostics
```
