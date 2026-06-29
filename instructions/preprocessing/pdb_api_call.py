#!/usr/bin/env python3
"""
Write matching PDB IDs (4-letter codes) to a text file.

Criteria:
- Contains at least one monosaccharide with a CCD code from the provided list.
- Entry resolution is 9 Å or better (i.e., resolution <= 9.0 Å).

This script uses the exact attributes from the PDB's own query builder
and sends requests in batches to avoid API size limits.
"""

import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
OUT_DIR = Path("Boltz_Data_Files")
OUT_FILE = OUT_DIR / "pdb_ids_with_glycans.txt"
BATCH_SIZE = 100  # Number of glycan CCD codes to query in each API call

# Resolution cutoff (Å): 9 Å or better means resolution <= 9.0
RESOLUTION_MAX_ANGSTROM = 9.0

# The full list of monosaccharide CCD codes.
ALLOWED_GLYCANS = [
    "05L", "07E", "0HX", "0LP", "0MK", "0NZ", "0UB", "0WK", "0XY", "0YT", "12E", "145", "147", "149", "14T", "15L", "16F", "16G", "16O", "17T", "18D", "18O", "1CF", "1GL", "1GN", "1S3", "1S4", "1SD", "1X4", "20S", "20X",
    "22O", "22S", "23V", "24S", "25E", "26O", "27C", "289", "291", "293", "2DG", "2DR", "2F8", "2FG", "2FL", "2GL", "2GS", "2H5", "2M5", "2M8", "2WP", "32O", "34V", "38J", "3DO", "3FM", "3HD", "3J3", "3J4", "3LJ", "3MG",
    "3MK", "3R3", "3S6", "3YW", "42D", "445", "44S", "46Z", "475", "491", "49A", "49S", "49T", "49V", "4AM", "4CQ", "4GL", "4GP", "4JA", "4N2", "4NN", "4QY", "4R1", "4SG", "4U0", "4U1", "4U2", "4UZ", "4V5", "50A",
    "510", "51N", "56N", "57S", "5DI", "5GF", "5GO", "5KQ", "5KV", "5L2", "5L3", "5LS", "5LT", "5N6", "5QP", "5TH", "5TJ", "5TK", "5TM", "604", "61J", "62I", "64K", "66O", "6BG", "6C2", "6GB", "6GP", "6GR",
    "6K3", "6KH", "6KL", "6KS", "6KU", "6KW", "6LS", "6LW", "6MJ", "6MN", "6PY", "6PZ", "6S2", "6UD", "6Y6", "6YR", "6ZC", "73E", "79J", "7CV", "7D1", "7GP", "7JZ", "7K2", "7K3", "7NU", "83Y", "89Y", "8B7", "8B9", "8EX", "8GA",
    "8GG", "8GP", "8LM", "8LR", "8OQ", "8PK", "8S0", "95Z", "96O", "9AM", "9C1", "9CD", "9GP", "9KJ", "9MR", "9OK", "9PG", "9QG", "9QZ", "9S7", "9SG", "9SJ", "9SM", "9SP", "9T1", "9T7", "9VP", "9WJ", "9WN", "9WZ", "9YW",
    "A0K", "A1Q", "A2G", "A5C", "A6P", "AAL", "ABD", "ABE", "ABF", "ABL", "AC1", "ACR", "ACX", "ADA", "AF1", "AFD", "AFO", "AFP", "AFR", "AGL", "AGR", "AH2", "AH8", "AHG", "AHM", "AHR", "AIG", "ALL", "ALX", "AMG", "AMN", "AMU",
    "AMV", "ANA", "AOG", "AQA", "ARA", "ARB", "ARI", "ARW", "ASC", "ASG", "ASO", "AXP", "AXR", "AY9", "AZC", "B0D", "B16", "B1H", "B1N", "B6D", "B7G", "B8D", "B9D", "BBK", "BBV", "BCD", "BCW", "BDF", "BDG", "BDP", "BDR", "BDZ",
    "BEM", "BFN", "BG6", "BG8", "BGC", "BGL", "BGN", "BGP", "BGS", "BHG", "BM3", "BM7", "BMA", "BMX", "BND", "BNG", "BNX", "BO1", "BOG", "BQY", "BS7", "BTG", "BTU", "BWG", "BXF", "BXX", "BXY", "BZD",
    "C3B", "C3G", "C3X", "C4B", "C4W", "C5X", "CBF", "CBI", "CBK", "CDR", "CE5", "CE6", "CE8", "CEG", "CEX", "CEY", "CEZ", "CGF", "CJB", "CKB", "CKP", "CNP", "CR1", "CR6", "CRA", "CT3", "CTO", "CTR", "CTT",
    "D0N", "D1M", "D5E", "D6G", "DAF", "DAG", "DAN", "DDA", "DDL", "DEG", "DEL", "DFR", "DFX", "DGO", "DGS", "DJB", "DJE", "DK4", "DKX", "DKZ", "DL6", "DLD", "DLF", "DLG", "DO8", "DOM", "DPC", "DQR", "DR2", "DR3", "DR5",
    "DRI", "DSR", "DT6", "DVC", "DYM", "E3M", "E5G", "EAG", "EBG", "EBQ", "EEN", "EEQ", "EGA", "EMP", "EMZ", "EPG", "EQP", "EQV", "ERE", "ERI", "ETT", "F1P", "F1X", "F55", "F58", "F6P", "FBP", "FCA", "FCB", "FCT", "FDP",
    "FDQ", "FFC", "FFX", "FIF", "FK9", "FKD", "FMF", "FMO", "FNG", "FNY", "FRU", "FSA", "FSI", "FSM", "FSR", "FSW", "FUB", "FUC", "FUF", "FUL", "FUY", "FVQ", "FX1", "FYJ", "G0S", "G16", "G1P", "G20", "G28", "G2F",
    "G3F", "G4D", "G4S", "G6D", "G6P", "G6S", "G7P", "G8Z", "GAA", "GAC", "GAD", "GAF", "GAL", "GAT", "GBH", "GC1", "GC4", "GC9", "GCB", "GCD", "GCN", "GCO", "GCS", "GCT", "GCU", "GCV", "GCW", "GDA", "GDL",
    "GE1", "GE3", "GFP", "GIV", "GL0", "GL1", "GL2", "GL4", "GL5", "GL6", "GL7", "GL9", "GLA", "GLC", "GLD", "GLF", "GLG", "GLO", "GLP", "GLS", "GLT", "GM0", "GMB", "GMH", "GMT", "GMZ", "GN1", "GN4", "GNS", "GNX",
    "GP0", "GP1", "GP4", "GPH", "GPK", "GPM", "GPO", "GPQ", "GPU", "GPV", "GPW", "GQ1", "GRF", "GRX", "GS1", "GS9", "GTK", "GTM", "GTR", "GU0", "GU1", "GU2", "GU3", "GU4", "GU5", "GU6", "GU8", "GU9", "GUF", "GUL", "GUP",
    "GUZ", "GXL", "GYE", "GYG", "GYP", "GYU", "GYV", "GZL", "H1M", "H1S", "H2P", "H53", "H6Q", "H6Z", "HBZ", "HD4", "HNV", "HNW", "HSG", "HSH", "HSJ", "HSQ", "HSX", "HSY", "HTG", "HTM", "I57", "IAB", "IDC", "IDF", "IDG", "IDR",
    "IDS", "IDU", "IDX", "IDY", "IEM", "IN1", "IPT", "ISD", "ISL", "ISX", "IXD", "J5B", "JFZ", "JHM", "JLT", "JS2", "JV4", "JVA", "JVS", "JZR", "K5B", "K99", "KBA", "KBG", "KD5", "KDA", "KDB", "KDD", "KDE", "KDF", "KDM", "KDN",
    "KDO", "KDR", "KFN", "KG1", "KGM", "KHP", "KME", "KO1", "KO2", "KOT", "KTU",
    "L1L", "L6S", "LAH", "LAK", "LAO", "LAT", "LB2", "LBS", "LBT", "LCN", "LDY", "LEC", "LFR", "LGC", "LGU", "LKA", "LKS", "LNV", "LOG", "LOX", "LRH", "LVO", "LVZ", "LXB", "LXC", "LXZ", "LZ0", "M1F", "M1P", "M2F", "M3N", "M55", "M6D",
    "M6P", "M7B", "M7P", "M8C", "MA1", "MA2", "MA3", "MA8", "MAF", "MAG", "MAL", "MAN", "MAT", "MAV", "MAW", "MBE", "MBF", "MBG", "MCU", "MDA", "MDP", "MFB", "MFU", "MG5", "MGC", "MGL", "MGS", "MJJ", "MLB", "MLR", "MMA", "MN0",
    "MNA", "MQG", "MQT", "MRH", "MRP", "MSX", "MTT", "MUB", "MUR", "MVP", "MXY", "MXZ", "MYG", "N1L", "N9S", "NA1", "NAA", "NAG", "NBG", "NBX", "NBY", "NDG", "NFG", "NG1", "NG6", "NGA", "NGC", "NGE", "NGK", "NGR", "NGS", "NGY", "NGZ", "NHF",
    "NLC", "NM6", "NM9", "NNG", "NPF", "NSQ", "NT1", "NTF", "NTO", "NTP", "NXD", "NYT",
    "O1G", "OAK", "OEL", "OI7", "OPM", "OSU", "OTG", "OTN", "OTU", "OX2", "P53", "P6P", "PA1", "PAV", "PDX", "PH5", "PKM", "PNA", "PNG", "PNJ", "PNW", "PPC", "PRP", "PSG", "PSV", "PUF", "PZU", "QIF", "QKH", "QPS", "R1P", "R1X", "R2B", "R2G",
    "RAE", "RAF", "RAM", "RAO", "RCD", "RER", "RF5", "RGG", "RHA", "RHC", "RI2", "RIB", "RIP", "RM4", "RP3", "RP5", "RP6", "RR7", "RRJ", "RRY", "RST", "RTG", "RTV", "RUG", "RUU", "RV7", "RVG", "RVM", "RWI", "RY7", "RZM", "S7P", "S81",
    "SA0", "SCG", "SCR", "SDY", "SEJ", "SF6", "SF9",
    "SFJ", "SFU", "SG4", "SG5", "SG6", "SG7", "SGA", "SGC", "SGD", "SGN", "SHB", "SHD", "SHG", "SIA", "SID", "SIO", "SIZ", "SLB", "SLM", "SLT", "SMD", "SN5", "SNG", "SOE", "SOG",
    "SOR", "SR1", "SSG", "STZ", "SUC", "SUP", "SUS", "SWE", "SZZ", "T68", "T6P", "T6T", "TA6", "TCB", "TCG", "TDG", "TEU", "TF0", "TFU", "TGA", "TGK", "TGR", "TGY", "TH1", "TMR",
    "TMX", "TNX", "TOA", "TOC", "TQY", "TRE", "TRV", "TS8", "TT7", "TTV", "TTZ", "TU4", "TUG", "TUJ", "TUP", "TUR", "TVD", "TVG", "TVM", "TVS", "TVV", "TVY", "TW7", "TWA", "TWD", "TWG", "TWJ", "TWY", "TXB", "TYV",
    "U1Y", "U2A", "U2D", "U63", "U8V", "U97", "U9A", "U9D", "U9G", "U9J", "U9M", "UAP", "UCD", "UDC", "UEA", "V3M", "V3P", "V71", "VG1", "VTB", "W9T", "WIA", "WOO", "WUN", "X0X", "X1P", "X1X", "X2F", "X6X", "XDX", "XGP",
    "XIL", "XLF", "XLS", "XMM", "XXM", "XXR", "XXX", "XYF", "XYL", "XYP", "XYS", "XYT", "XYZ", "YIO", "YJM", "YKR", "YO5", "YX0", "YX1", "YYB", "YYH", "YYJ", "YYK", "YYM", "YYQ", "YZ0", "Z0F", "Z15", "Z16", "Z2D", "Z2T", "Z3K", "Z3L", "Z3Q", "Z3U",
    "Z4K", "Z4R", "Z4S", "Z4U", "Z4V", "Z4W", "Z4Y", "Z57", "Z5J", "Z5L", "Z61", "Z6H", "Z6J", "Z6W", "Z8H", "Z8T", "Z9D", "Z9E", "Z9H", "Z9K", "Z9L", "Z9M", "Z9N", "Z9W", "ZB0", "ZB1", "ZB2", "ZB3", "ZCD", "ZCZ", "ZD0", "ZDC", "ZDO", "ZEE", "ZEL", "ZGE", "ZMR"
]

def fetch_pdb_ids_for_batch(glycan_batch):
    """Fetch PDB IDs that (1) contain at least one of the supplied CCD codes AND
       (2) meet either resolution threshold: X-ray <= RESOLUTION_MAX_ANGSTROM OR
       EM <= RESOLUTION_MAX_ANGSTROM.
    """
    payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "group",
                    "logical_operator": "or",
                    "nodes": [
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_entry_info.diffrn_resolution_high.value",
                                "operator": "less_or_equal",
                                "value": RESOLUTION_MAX_ANGSTROM
                            }
                        },
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "em_3d_reconstruction.resolution",
                                "operator": "less_or_equal",
                                "value": RESOLUTION_MAX_ANGSTROM
                            }
                        }
                    ]
                },
                {
                    "type": "terminal",
                    "service": "text_chem",
                    "parameters": {
                        "attribute": "rcsb_chem_comp_container_identifiers.comp_id",
                        "operator": "in",
                        "value": glycan_batch
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True,
            "results_content_type": ["experimental"],
            "scoring_strategy": "combined"
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req = Request(SEARCH_URL, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"})

    try:
        with urlopen(req, timeout=90) as resp:
            if resp.status == 204:
                return []
            raw = resp.read()
            js = json.loads(raw.decode("utf-8"))
            return js.get("result_set", [])
    except HTTPError as e:
        print(f"  -> HTTP Error {e.code} for batch. Skipping this batch.")
        try:
            print(f"  -> Server response: {e.read().decode(errors='ignore')[:200]}")
        except Exception:
            pass
        return []
    except URLError as e:
        print(f"  -> Network Error for batch: {e.reason}. Skipping this batch.")
        return []



def main():
    OUT_DIR.mkdir(exist_ok=True)

    all_pdb_ids = set()
    num_batches = (len(ALLOWED_GLYCANS) + BATCH_SIZE - 1) // BATCH_SIZE

    print(
        f"Querying {len(ALLOWED_GLYCANS)} glycan CCD codes in {num_batches} batches of up to {BATCH_SIZE} each.\n"
        f"Resolution filter: <= {RESOLUTION_MAX_ANGSTROM:.1f} Å (9 Å or better)."
    )

    for i in range(0, len(ALLOWED_GLYCANS), BATCH_SIZE):
        batch = ALLOWED_GLYCANS[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        print(f"Fetching batch {batch_num}/{num_batches}...")

        # The API returns a list of dictionaries, e.g., [{'identifier': '1ABC', 'score': 1.0}]
        results_from_batch = fetch_pdb_ids_for_batch(batch)

        if results_from_batch:
            # Extract the 'identifier' string from each dictionary
            ids_from_batch = {result['identifier'] for result in results_from_batch}

            count = len(ids_from_batch)
            new_ids = len(ids_from_batch - all_pdb_ids)
            print(f"  -> Success! Found {count} total IDs in this batch, {new_ids} of which are new.")
            all_pdb_ids.update(ids_from_batch)
        else:
            print("  -> No IDs found in this batch or an error occurred.")

        # Pause briefly between requests to be respectful to the API server
        if batch_num < num_batches:
            time.sleep(0.5)

    if all_pdb_ids:
        sorted_ids = sorted(list(all_pdb_ids))
        OUT_FILE.write_text("\n".join(sorted_ids) + "\n", encoding="utf-8")
        print(f"\nSuccess! Wrote a total of {len(sorted_ids)} unique PDB IDs to {OUT_FILE.resolve()}")
    else:
        OUT_FILE.touch()
        print(f"\nNo PDB IDs matched the criteria. Empty file created at {OUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
