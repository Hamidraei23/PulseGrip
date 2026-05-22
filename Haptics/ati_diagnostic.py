"""
ATI diagnostic — raw counts and force in Newtons, NO tare.
Watch if Fz_raw changes when you press the sensor against the table.
ATI 192.168.4.141 is configured in Newtons (cpf=1000000).
"""
import time
from sensors import ATIAxia80

ati = ATIAxia80(host="192.168.4.141", cpf=1000000.0, cpt=1000000.0)
ati.start_streaming()

print("Reading raw counts — press ATI flat against table and watch Fz_raw change")
print(f"{'Fx_raw':>12} {'Fy_raw':>12} {'Fz_raw':>12}  |  {'Fx_N':>8} {'Fy_N':>8} {'Fz_N':>8}")
print("-" * 80)

while True:
    ati.drain_buffer()
    counts = ati._recv_counts()
    fx, fy, fz = counts[:3]
    print(f"{fx:>12.0f} {fy:>12.0f} {fz:>12.0f}  |  "
          f"{fx/ati._cpf:>8.3f} {fy/ati._cpf:>8.3f} {fz/ati._cpf:>8.3f}")
    time.sleep(0.05)
