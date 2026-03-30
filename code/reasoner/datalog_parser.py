"""
Datalog XML Parser
Extracts usable observations from the Canon Colorado M-series datalog XML file.

Extracts:
1. Ink system counters (per-color consumption, pump operating times, etc.)
2. Ink system parameters (expiry dates/states, safe mode, etc.)
3. Recent error/warning events from event history (filtered to 02_INK MRI)
4. System-level counters (total usage, etc.)
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class InkColorData:
    """Per-color ink data extracted from snapshot."""
    color: str  # C, M, Y, K, W
    motor_running_time_s: float = 0.0
    inserted_cartridges: int = 0
    total_loaded_ink_ml: float = 0.0
    expiry_state: int = 0  # 0=ok, 1=expired
    ink_safe_mode: int = 0
    expiry_date_raw: str = ""


@dataclass
class PumpData:
    """Per-pump data extracted from snapshot."""
    pump_id: str  # INK_PUMP_2_MO through INK_PUMP_6_MO
    operating_time_pump_ds: int = 0  # deciseconds
    operating_time_tube_ds: int = 0  # deciseconds
    mark_ink_tube_empty: int = 0


@dataclass
class ErrorEvent:
    """An error/warning event from the history."""
    date: str
    counter: int
    error_code: str
    mri: str
    error_class: str  # Warning, ORE, MRE
    details: str = ""
    
    @property
    def component(self) -> str:
        """Extract the specific component from MRI path.
        e.g., //system/printer/02_INK/C -> C
              //system/printer/02_INK/INK_PUMP_3_MO -> INK_PUMP_3_MO
        """
        parts = self.mri.split("/")
        if len(parts) > 0:
            return parts[-1]
        return ""


@dataclass
class DatalogExtraction:
    """All observations extracted from a datalog XML."""
    # Machine identity
    system_type: str = ""
    system_id: str = ""
    master_usage_counter: int = 0
    snapshot_date: str = ""
    
    # System-level counters
    system_counters: Dict[str, int] = field(default_factory=dict)
    
    # Ink system data
    ink_colors: Dict[str, InkColorData] = field(default_factory=dict)
    pumps: Dict[str, PumpData] = field(default_factory=dict)
    
    # Event history (ink-related)
    ink_errors: List[ErrorEvent] = field(default_factory=list)
    
    # All errors (for context)
    all_recent_errors: List[ErrorEvent] = field(default_factory=list)


class DatalogParser:
    """Parses Canon Colorado M-series datalog XML files."""

    # MRI paths that indicate ink system components
    INK_MRI_PREFIX = "//system/printer/02_INK"
    
    # Color codes
    INK_COLORS = {"C", "M", "Y", "K", "W"}
    
    # Counter descriptions we care about at system level
    RELEVANT_SYSTEM_COUNTERS = {
        "218501": "engine_initializations",
        "218502": "engine_uptime_min",
        "218503": "number_of_runs",
        "218510": "total_printed_area_m2",
        "218519": "total_ink_consumption_ml",
        "218512": "ink_consumption_cyan_ml",
        "218513": "ink_consumption_magenta_ml",
        "218514": "ink_consumption_yellow_ml",
        "218515": "ink_consumption_black_ml",
        "218516": "ink_consumption_white_ml",
        "48501": "cold_startups",
        "48503": "warm_startups",
        "78014": "manual_maintenance_printheads",
        "78015": "auto_maintenance_printheads",
        "78042": "total_spit_actions",
    }

    def __init__(self, lookback_days: int = 30):
        """
        Args:
            lookback_days: How far back to look for relevant events from snapshot date.
        """
        self.lookback_days = lookback_days

    def parse(self, xml_path: str) -> DatalogExtraction:
        """Parse a datalog XML file and return extracted observations."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        result = DatalogExtraction()
        
        # Extract system info
        self._parse_system_info(root, result)
        
        # Find current snapshot
        current_snapshot = self._find_current_snapshot(root)
        if current_snapshot is None:
            raise ValueError("No current snapshot found in datalog")
        
        result.snapshot_date = current_snapshot.get("date", "")
        
        # Extract counters and parameters from snapshot
        self._parse_snapshot(current_snapshot, result)
        
        # Extract event history
        self._parse_events(root, result)
        
        return result

    def _strip_ns(self, tag: str) -> str:
        """Strip XML namespace from tag."""
        return tag.split("}")[-1] if "}" in tag else tag

    def _parse_system_info(self, root, result: DatalogExtraction):
        """Extract system identification."""
        for child in root:
            if self._strip_ns(child.tag) == "systemInfo":
                for item in child:
                    tag = self._strip_ns(item.tag)
                    if tag == "systemType":
                        result.system_type = item.get("value", "")
                    elif tag == "systemIdentification":
                        result.system_id = item.get("value", "")
                    elif tag == "masterUsageCounter":
                        result.master_usage_counter = int(item.get("value", "0"))
                    elif tag == "parameter":
                        name = item.get("name", "")
                        if name in self.RELEVANT_SYSTEM_COUNTERS:
                            friendly = self.RELEVANT_SYSTEM_COUNTERS[name]
                            result.system_counters[friendly] = int(item.get("value", "0"))
                break

    def _find_current_snapshot(self, root) -> Optional[ET.Element]:
        """Find the snapshot marked current='true'."""
        for child in root:
            if self._strip_ns(child.tag) == "snapshot" and child.get("current") == "true":
                return child
        return None

    def _parse_snapshot(self, snapshot: ET.Element, result: DatalogExtraction):
        """Extract ink system data from the current snapshot."""
        # Navigate: snapshot > resource[system] > resource[printer]
        for sys_resource in snapshot:
            if self._strip_ns(sys_resource.tag) == "resource" and sys_resource.get("name") == "system":
                for printer_resource in sys_resource:
                    if self._strip_ns(printer_resource.tag) == "resource" and printer_resource.get("name") == "printer":
                        self._parse_printer_resource(printer_resource, result)
                        break
                break

    def _parse_printer_resource(self, printer: ET.Element, result: DatalogExtraction):
        """Extract data from printer resource, focusing on 02_INK."""
        # Get system-level printer counters
        for child in printer:
            tag = self._strip_ns(child.tag)
            if tag == "counters":
                for counter in child:
                    name = counter.get("name", "")
                    if name in self.RELEVANT_SYSTEM_COUNTERS:
                        friendly = self.RELEVANT_SYSTEM_COUNTERS[name]
                        result.system_counters[friendly] = int(counter.get("value", "0"))
            elif tag == "resource" and child.get("name") == "02_INK":
                self._parse_ink_resource(child, result)

    def _parse_ink_resource(self, ink_resource: ET.Element, result: DatalogExtraction):
        """Extract per-color and per-pump data from 02_INK resource."""
        for child in ink_resource:
            if self._strip_ns(child.tag) != "resource":
                continue
            
            name = child.get("name", "")
            
            if name in self.INK_COLORS:
                # Per-color ink data
                color_data = InkColorData(color=name)
                self._parse_color_resource(child, color_data)
                result.ink_colors[name] = color_data
            
            elif name.startswith("INK_PUMP_"):
                # Per-pump data
                pump_data = PumpData(pump_id=name)
                self._parse_pump_resource(child, pump_data)
                result.pumps[name] = pump_data

    def _parse_color_resource(self, resource: ET.Element, color: InkColorData):
        """Extract counters and parameters for a single ink color."""
        for child in resource:
            tag = self._strip_ns(child.tag)
            if tag == "counters":
                for counter in child:
                    desc = counter.get("description", "").lower()
                    val = counter.get("value", "0")
                    if "motor" in desc and "running" in desc:
                        color.motor_running_time_s = float(val)
                    elif "inserted cartridges" in desc and "current" not in desc:
                        color.inserted_cartridges = int(val)
                    elif "total loaded ink" in desc:
                        color.total_loaded_ink_ml = float(val)
            elif tag == "parameters":
                for param in child:
                    desc = param.get("description", "").lower()
                    val = param.get("value", "")
                    if "expiry date" in desc:
                        color.expiry_date_raw = val
                    elif "expiry state" in desc:
                        color.expiry_state = int(val)
                    elif "safe mode" in desc:
                        color.ink_safe_mode = int(val)

    def _parse_pump_resource(self, resource: ET.Element, pump: PumpData):
        """Extract counters and parameters for a single pump."""
        for child in resource:
            tag = self._strip_ns(child.tag)
            if tag == "counters":
                for counter in child:
                    desc = counter.get("description", "").lower()
                    val = int(counter.get("value", "0"))
                    if "ink pump" in desc:
                        pump.operating_time_pump_ds = val
                    elif "pump tube" in desc:
                        pump.operating_time_tube_ds = val
            elif tag == "parameters":
                for param in child:
                    desc = param.get("description", "").lower()
                    if "mark ink tube" in desc:
                        pump.mark_ink_tube_empty = int(param.get("value", "0"))

    def _parse_events(self, root, result: DatalogExtraction):
        """Extract ink-related error/warning events from history."""
        events_elem = None
        for child in root:
            if self._strip_ns(child.tag) == "events":
                events_elem = child
                break
        
        if events_elem is None:
            return
        
        history_elem = None
        for child in events_elem:
            if self._strip_ns(child.tag) == "history":
                history_elem = child
                break
        
        if history_elem is None:
            return
        
        # Determine lookback window
        snapshot_dt = None
        if result.snapshot_date:
            try:
                snapshot_dt = datetime.fromisoformat(result.snapshot_date)
            except ValueError:
                pass
        
        cutoff = None
        if snapshot_dt:
            cutoff = snapshot_dt - timedelta(days=self.lookback_days)
        
        for child in history_elem:
            tag = self._strip_ns(child.tag)
            if tag != "error":
                continue
            
            date_str = child.get("date", "")
            
            # Apply lookback filter if we have a cutoff
            if cutoff and date_str:
                try:
                    event_dt = datetime.fromisoformat(date_str)
                    if event_dt < cutoff:
                        continue
                except ValueError:
                    pass
            
            error_code = child.get("errorCode", "")
            mri = child.get("mri", "")
            error_class = child.get("class", "")
            counter_val = int(child.get("counter", "0"))
            
            # Extract ErrorDetails from child elements
            details = ""
            for subchild in child:
                if self._strip_ns(subchild.tag) == "parameter" and subchild.get("name") == "ErrorDetails":
                    details = subchild.get("value", "")
            
            event = ErrorEvent(
                date=date_str,
                counter=counter_val,
                error_code=error_code,
                mri=mri,
                error_class=error_class,
                details=details,
            )
            
            # Filter for ink-related
            if self.INK_MRI_PREFIX in mri:
                result.ink_errors.append(event)
            
            result.all_recent_errors.append(event)

    def extract_observations(self, extraction: DatalogExtraction, kb) -> List[Tuple[str, str]]:
        """
        Convert raw extracted data into (obs_id, obs_value) pairs
        that map to the observation registry.
        
        Args:
            extraction: Parsed datalog data
            kb: KnowledgeBase instance for obs_id lookup
            
        Returns:
            List of (obs_id, obs_value) tuples ready for the reasoner
        """
        observations = []
        
        # 1. Map error codes to observations
        # Deduplicate: take the most recent occurrence of each error code
        seen_codes = set()
        for event in reversed(extraction.ink_errors):
            code = event.error_code
            if code not in seen_codes:
                seen_codes.add(code)
                obs_id = kb.obs_by_error_code.get(code)
                if obs_id:
                    # CPT uses "1" for present, "0" for absent
                    observations.append((obs_id, "1"))
        
        # 2. Map counter-based observations
        # Pump tube operating times -> bucketed observations
        for pump_id, pump in extraction.pumps.items():
            tube_hours = pump.operating_time_tube_ds / 36000  # deciseconds to hours
            # Find matching observation in registry
            for obs_id, obs in kb.observations.items():
                if obs.obs_type == "counter" and pump_id.lower() in obs.obs_label.lower():
                    if obs.bucketing_thresholds:
                        bucket = self._bucket_value(tube_hours, obs.bucketing_thresholds)
                        observations.append((obs_id, bucket))
                    break
        
        # 3. Expiry state observations
        for color, data in extraction.ink_colors.items():
            if data.expiry_state == 1:
                # Look for expiry-related observation
                for obs_id, obs in kb.observations.items():
                    if obs.obs_type == "parameter" and "expir" in obs.obs_label.lower() and color.lower() in obs.obs_label.lower():
                        observations.append((obs_id, "expired"))
                        break
        
        return observations

    def _bucket_value(self, value: float, thresholds: Dict) -> str:
        """Bucket a numeric value using threshold definitions."""
        if not thresholds:
            return str(value)
        
        # Expected format: {"low": X, "medium": Y, "high": Z}
        # or {"threshold_hours": X} for simple binary
        if "high" in thresholds:
            if value >= thresholds["high"]:
                return "high"
            elif value >= thresholds.get("medium", thresholds["high"] / 2):
                return "medium"
            else:
                return "low"
        elif "threshold_hours" in thresholds:
            return "above" if value >= thresholds["threshold_hours"] else "below"
        
        return str(value)

    def summary(self, extraction: DatalogExtraction) -> str:
        """Human-readable summary of extracted data."""
        lines = [
            f"Datalog Summary",
            f"  System: {extraction.system_type} (ID: {extraction.system_id})",
            f"  Snapshot: {extraction.snapshot_date}",
            f"  Master counter: {extraction.master_usage_counter}",
            f"",
            f"Ink Colors:",
        ]
        for color, data in sorted(extraction.ink_colors.items()):
            lines.append(
                f"  {color}: loaded={data.total_loaded_ink_ml}mL, "
                f"cartridges={data.inserted_cartridges}, "
                f"expiry_state={'EXPIRED' if data.expiry_state else 'OK'}, "
                f"safe_mode={'ON' if data.ink_safe_mode else 'OFF'}"
            )
        
        lines.append(f"\nPumps:")
        for pump_id, data in sorted(extraction.pumps.items()):
            tube_hrs = data.operating_time_tube_ds / 36000
            pump_hrs = data.operating_time_pump_ds / 36000
            lines.append(
                f"  {pump_id}: pump={pump_hrs:.1f}h, tube={tube_hrs:.1f}h, "
                f"tube_empty={'YES' if data.mark_ink_tube_empty else 'NO'}"
            )
        
        lines.append(f"\nInk Errors (last {len(extraction.ink_errors)} in lookback window):")
        for event in extraction.ink_errors[-10:]:
            lines.append(
                f"  {event.date} | {event.error_class} {event.error_code} | "
                f"{event.component} | {event.details}"
            )
        
        return "\n".join(lines)
