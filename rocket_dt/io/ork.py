"""OpenRocket (.ork) parser.

Produces a tree of plain dataclasses describing the airframe exactly as stored in
the file. Nothing is computed here except unit conversion; positions are resolved
by :func:`resolve_positions` *after* user overrides have been applied, so an
override of e.g. a tube length correctly shifts everything behind it.

Axial stations (``x_top``) are measured from the nose tip, positive aft, in metres,
which is OpenRocket's convention.

Unsupported features raise :class:`OrkError` instead of being silently ignored.
"""

from __future__ import annotations

import io
import math
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


class OrkError(ValueError):
    pass


# Surface roughness per OpenRocket finish (m)
FINISH_ROUGHNESS = {
    "rough": 500e-6,
    "unfinished": 150e-6,
    "normal": 60e-6,
    "smooth": 20e-6,
    "finishpolished": 2e-6,
    "polished": 2e-6,
    "mirror": 0.0,
}

# OpenRocket fin cross-section volume factors (affect fin mass only)
CROSS_SECTION_VOLUME_FACTOR = {"square": 1.0, "rounded": 0.99, "airfoil": 0.85}


@dataclass
class Material:
    name: str
    kind: str  # "bulk" (kg/m^3) | "surface" (kg/m^2) | "line" (kg/m)
    density: float


@dataclass
class Component:
    kind: str
    name: str
    id: str
    position_method: str = "after"
    position_offset: float = 0.0
    length: float = 0.0
    material: Material | None = None
    finish: str = "normal"
    override_mass: float | None = None
    override_cg: float | None = None  # from component top, m
    children: list["Component"] = field(default_factory=list)
    parent: "Component | None" = field(default=None, repr=False, compare=False)
    x_top: float = math.nan  # resolved absolute station of component top

    @property
    def path(self) -> str:
        """'/'-separated names from the top level down. Siblings sharing a name are
        disambiguated as 'Name#1', 'Name#2' (in file order)."""
        parts = []
        c: Component | None = self
        while c is not None and c.kind != "stage":
            seg = c.name.strip()
            if c.parent is not None:
                same = [s for s in c.parent.children if s.name.strip() == seg]
                if len(same) > 1:
                    seg += f"#{next(i for i, s in enumerate(same) if s is c) + 1}"
            parts.append(seg)
            c = c.parent
        return "/".join(reversed(parts))

    def walk(self):
        yield self
        for ch in self.children:
            yield from ch.walk()


@dataclass
class Shoulder:
    radius: float = 0.0
    length: float = 0.0
    thickness: float = 0.0
    capped: bool = False


@dataclass
class SymmetricComponent(Component):
    """Nose cone, body tube or transition (axisymmetric shell)."""
    fore_radius: float = 0.0
    aft_radius: float = 0.0
    thickness: float = 0.0
    filled: bool = False
    shape: str = "conical"
    shape_param: float = 0.0
    fore_shoulder: Shoulder = field(default_factory=Shoulder)
    aft_shoulder: Shoulder = field(default_factory=Shoulder)
    motor_overhang: float | None = None  # set if this is a motor mount


@dataclass
class FinSet(Component):
    fin_count: int = 0
    planform: list[tuple[float, float]] = field(default_factory=list)  # (x from root LE, y span)
    thickness: float = 0.0
    cross_section: str = "square"
    cant: float = 0.0  # rad
    angle_offset: float = 0.0  # rad, roll position of first fin


@dataclass
class MassComponent(Component):
    mass: float = 0.0
    radius: float = 0.0  # packed radius
    radial_position: float = 0.0


@dataclass
class RingComponent(Component):
    """Inner tube, coupler, centering ring, bulkhead, engine block."""
    outer_radius: float = 0.0
    inner_radius: float = 0.0
    radial_position: float = 0.0
    motor_overhang: float | None = None


@dataclass
class OrkRocket:
    name: str
    components: list[Component]  # direct children of the (single) stage
    reference_type: str
    custom_reference_length: float | None
    ork_version: str
    creator: str

    def walk(self):
        for c in self.components:
            yield from c.walk()

    def find(self, path: str) -> Component:
        """Find a component by '/'-separated path of names (whitespace-insensitive),
        by a unique bare name, or by its OpenRocket id."""
        key = "/".join(p.strip() for p in path.split("/"))
        matches = [c for c in self.walk() if c.path == key or c.id == path]
        if not matches and "/" not in key:
            matches = [c for c in self.walk() if c.name.strip() == key]
        if len(matches) != 1:
            avail = "\n  ".join(c.path for c in self.walk())
            what = "not found" if not matches else f"ambiguous ({len(matches)} matches)"
            raise OrkError(f"Component '{path}' {what}. Available paths:\n  {avail}")
        return matches[0]


# ----------------------------------------------------------------------------- parsing


def _read_xml(path: Path) -> ET.Element:
    data = path.read_bytes()
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = [n for n in z.namelist() if n.endswith(".ork") or n.endswith(".xml")] or z.namelist()
            if len(names) != 1:
                raise OrkError(f"{path}: expected one .ork inside the archive, found {names}")
            data = z.read(names[0])
            if data[:2] == b"PK":  # a zipped .ork that is itself a zip
                with zipfile.ZipFile(io.BytesIO(data)) as z2:
                    data = z2.read(z2.namelist()[0])
    return ET.fromstring(data)


def _f(el: ET.Element, tag: str, default: float | None = None) -> float:
    t = el.findtext(tag)
    if t is None:
        if default is None:
            raise OrkError(f"<{el.tag}> '{el.findtext('name')}' is missing <{tag}>")
        return default
    t = t.strip()
    if t.startswith("auto"):
        t = t.split()[-1]
    return float(t)


def _b(el: ET.Element, tag: str) -> bool:
    return (el.findtext(tag) or "false").strip().lower() == "true"


def _material(el: ET.Element, tag: str = "material") -> Material | None:
    m = el.find(tag)
    if m is None:
        return None
    return Material(name=(m.text or "").strip(), kind=m.get("type", "bulk"), density=float(m.get("density")))


def _common(el: ET.Element, cls, kind: str, **kw) -> Component:
    pos = el.find("axialoffset")
    if pos is None:
        pos = el.find("position")
    method = pos.get("method", pos.get("type", "after")) if pos is not None else "after"
    offset = float(pos.text) if pos is not None and pos.text else 0.0

    for tag in ("overridecd", "overridesubcomponents", "overridesubcomponentsmass",
                "overridesubcomponentscg", "overridesubcomponentscd"):
        if _b(el, tag):
            raise OrkError(f"'{el.findtext('name')}': <{tag}> is not supported "
                           "(express it as an explicit override in the YAML config instead)")
    om = el.findtext("overridemass")
    ocg = el.findtext("overridecg")
    c = cls(kind=kind, name=el.findtext("name") or kind, id=el.findtext("id") or "",
            position_method=method, position_offset=offset,
            material=_material(el), finish=(el.findtext("finish") or "normal").strip(),
            override_mass=float(om) if om else None, override_cg=float(ocg) if ocg else None, **kw)
    if c.finish not in FINISH_ROUGHNESS:
        raise OrkError(f"'{c.name}': unknown finish '{c.finish}'")
    return c


def _shoulder(el: ET.Element, prefix: str) -> Shoulder:
    return Shoulder(radius=_f(el, f"{prefix}shoulderradius", 0.0), length=_f(el, f"{prefix}shoulderlength", 0.0),
                    thickness=_f(el, f"{prefix}shoulderthickness", 0.0), capped=_b(el, f"{prefix}shouldercapped"))


def _motor_overhang(el: ET.Element) -> float | None:
    mm = el.find("motormount")
    if mm is None:
        return None
    return _f(mm, "overhang", 0.0)


def _parse_component(el: ET.Element) -> Component:
    tag = el.tag
    if tag == "nosecone":
        if _b(el, "isflipped"):
            raise OrkError("Flipped nose cones are not supported")
        c = _common(el, SymmetricComponent, "nosecone", length=_f(el, "length"),
                    fore_radius=0.0, aft_radius=_f(el, "aftradius"), thickness=_f(el, "thickness"),
                    filled=_b(el, "filled"), shape=el.findtext("shape").strip(),
                    shape_param=_f(el, "shapeparameter", 0.0), aft_shoulder=_shoulder(el, "aft"))
    elif tag == "bodytube":
        r = _f(el, "radius")
        c = _common(el, SymmetricComponent, "bodytube", length=_f(el, "length"), fore_radius=r, aft_radius=r,
                    thickness=_f(el, "thickness"), filled=_b(el, "filled"), motor_overhang=_motor_overhang(el))
    elif tag == "transition":
        c = _common(el, SymmetricComponent, "transition", length=_f(el, "length"),
                    fore_radius=_f(el, "foreradius"), aft_radius=_f(el, "aftradius"),
                    thickness=_f(el, "thickness"), filled=_b(el, "filled"),
                    shape=el.findtext("shape").strip(), shape_param=_f(el, "shapeparameter", 0.0),
                    fore_shoulder=_shoulder(el, "fore"), aft_shoulder=_shoulder(el, "aft"))
    elif tag in ("trapezoidfinset", "ellipticalfinset", "freeformfinset"):
        c = _parse_finset(el)
    elif tag == "masscomponent":
        c = _common(el, MassComponent, "masscomponent", length=_f(el, "packedlength"), mass=_f(el, "mass"),
                    radius=_f(el, "packedradius"), radial_position=_f(el, "radialposition", 0.0))
    elif tag in ("parachute", "streamer", "shockcord"):
        c = _parse_recovery(el)
    elif tag in ("innertube", "tubecoupler", "engineblock"):
        ro = _f(el, "outerradius")
        if el.find("clusterconfiguration") is not None and el.findtext("clusterconfiguration").strip() != "single":
            raise OrkError(f"'{el.findtext('name')}': motor clusters are not supported")
        c = _common(el, RingComponent, tag, length=_f(el, "length"), outer_radius=ro,
                    inner_radius=ro - _f(el, "thickness"), radial_position=_f(el, "radialposition", 0.0),
                    motor_overhang=_motor_overhang(el))
    elif tag in ("centeringring", "bulkhead"):
        c = _common(el, RingComponent, tag, length=_f(el, "length"), outer_radius=_f(el, "outerradius"),
                    inner_radius=_f(el, "innerradius", 0.0), radial_position=_f(el, "radialposition", 0.0))
    else:
        raise OrkError(f"Unsupported OpenRocket component <{tag}> ('{el.findtext('name')}')")

    sub = el.find("subcomponents")
    if sub is not None:
        for ch_el in sub:
            ch = _parse_component(ch_el)
            ch.parent = c
            c.children.append(ch)
    return c


def _parse_finset(el: ET.Element) -> FinSet:
    tag = el.tag
    name = el.findtext("name")
    if _f(el, "filletradius", 0.0) > 0:
        raise OrkError(f"'{name}': fin fillets are not supported")
    if _f(el, "tabheight", 0.0) > 0 and _f(el, "tablength", 0.0) > 0:
        raise OrkError(f"'{name}': fin tabs are not supported (add their mass as an override)")
    ro = el.find("radiusoffset")
    if ro is not None and (ro.get("method", "surface") != "surface" or abs(float(ro.text)) > 1e-9):
        raise OrkError(f"'{name}': only fins mounted on the body surface are supported")

    if tag == "trapezoidfinset":
        cr, ct = _f(el, "rootchord"), _f(el, "tipchord")
        sw, s = _f(el, "sweeplength"), _f(el, "height")
        planform = [(0.0, 0.0), (sw, s), (sw + ct, s), (cr, 0.0)]
        root = cr
    elif tag == "ellipticalfinset":
        cr, s = _f(el, "rootchord"), _f(el, "height")
        n = 48
        lead = [(cr / 2 - cr / 2 * math.cos(math.asin(min(1.0, y / s))) if y < s else cr / 2, y)
                for y in (s * i / n for i in range(n + 1))]
        trail = [(cr - x, y) for (x, y) in reversed(lead)]
        planform = lead + trail
        root = cr
    else:
        pts = el.find("finpoints")
        planform = [(float(p.get("x")), float(p.get("y"))) for p in pts]
        root = planform[-1][0] - planform[0][0]

    count = int(el.findtext("fincount") or el.findtext("instancecount"))
    cant = math.radians(_f(el, "cant", 0.0))
    angle = math.radians(_f(el, "angleoffset", 0.0) + _f(el, "rotation", 0.0))
    return _common(el, FinSet, "finset", length=root, fin_count=count, planform=planform,
                   thickness=_f(el, "thickness"), cross_section=(el.findtext("crosssection") or "square").strip(),
                   cant=cant, angle_offset=angle)


def _parse_recovery(el: ET.Element) -> MassComponent:
    tag = el.tag
    mat = _material(el)
    if tag == "parachute":
        d = _f(el, "diameter")
        mass = math.pi * d * d / 4.0 * mat.density
        lm = _material(el, "linematerial")
        if lm is not None:
            mass += int(el.findtext("linecount") or 0) * _f(el, "linelength", 0.0) * lm.density
    elif tag == "streamer":
        mass = _f(el, "striplength") * _f(el, "stripwidth") * mat.density
    else:
        mass = _f(el, "cordlength") * mat.density
    c = _common(el, MassComponent, tag, length=_f(el, "packedlength"), mass=mass,
                radius=_f(el, "packedradius"), radial_position=_f(el, "radialposition", 0.0))
    c.material = None  # mass already computed from the surface/line material
    return c


def load_ork(path: str | Path) -> OrkRocket:
    path = Path(path)
    root = _read_xml(path)
    if root.tag != "openrocket":
        raise OrkError(f"{path} is not an OpenRocket file")
    rk = root.find("rocket")
    stages = rk.find("subcomponents").findall("stage")
    if len(stages) != 1:
        raise OrkError(f"Only single-stage rockets are supported (found {len(stages)} stages)")
    stage_el = stages[0]
    stage = Component(kind="stage", name=stage_el.findtext("name") or "stage", id=stage_el.findtext("id") or "")
    for el in stage_el.find("subcomponents"):
        if el.tag in ("podset", "parallelstage"):
            raise OrkError("Pods / boosters are not supported")
        c = _parse_component(el)
        c.parent = stage
        stage.children.append(c)

    ref_type = (rk.findtext("referencetype") or "maximum").strip()
    custom = rk.findtext("customreference")
    return OrkRocket(name=rk.findtext("name") or "rocket", components=stage.children,
                     reference_type=ref_type, custom_reference_length=float(custom) if custom else None,
                     ork_version=root.get("version", "?"), creator=root.get("creator", "?"))


# ----------------------------------------------------------------------------- positions


def resolve_positions(rocket: OrkRocket) -> None:
    """Fill in ``x_top`` (absolute station of each component's fore end)."""
    x = 0.0
    for c in rocket.components:
        if c.position_method in ("after", "top"):  # stage children are laid out in sequence
            c.x_top = x + c.position_offset
        elif c.position_method == "absolute":
            c.x_top = c.position_offset
        else:
            raise OrkError(f"'{c.name}': axial method '{c.position_method}' not supported for top-level components")
        x = c.x_top + c.length
        _resolve_children(c)


def _resolve_children(parent: Component) -> None:
    prev: Component | None = None
    for c in parent.children:
        m, off = c.position_method, c.position_offset
        if m == "top":
            c.x_top = parent.x_top + off
        elif m == "middle":
            c.x_top = parent.x_top + (parent.length - c.length) / 2.0 + off
        elif m == "bottom":
            c.x_top = parent.x_top + parent.length - c.length + off
        elif m == "absolute":
            c.x_top = off
        elif m == "after":
            base = (prev.x_top + prev.length) if prev is not None else parent.x_top
            c.x_top = base + off
        else:
            raise OrkError(f"'{c.name}': unknown axial method '{m}'")
        prev = c
        _resolve_children(c)


def rocket_length(rocket: OrkRocket) -> float:
    return max(c.x_top + c.length for c in rocket.components)


def reference_length(rocket: OrkRocket) -> float:
    """Reference diameter per the file's <referencetype>."""
    bodies = [c for c in rocket.components if isinstance(c, SymmetricComponent)]
    if rocket.reference_type == "maximum":
        return 2.0 * max(max(c.fore_radius, c.aft_radius) for c in bodies)
    if rocket.reference_type == "nosecone":
        return 2.0 * bodies[0].aft_radius
    if rocket.reference_type == "custom" and rocket.custom_reference_length:
        return rocket.custom_reference_length
    raise OrkError(f"Unsupported reference type '{rocket.reference_type}'")
