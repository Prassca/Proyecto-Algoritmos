"""DetectaGrietas - servidor local.
Uso:  python servidor.py          (puerto 8000)
      python servidor.py 8080     (otro puerto)
Deja index.html y pesos.bin en la misma carpeta que este archivo.
Abre en el celular (misma red WiFi) la direccion que se imprime.
"""
import http.server, socketserver, socket, sys, os, functools

PUERTO = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
CARPETA = os.path.dirname(os.path.abspath(__file__))

for f in ("index.html", "pesos.bin"):
    if not os.path.exists(os.path.join(CARPETA, f)):
        sys.exit(f"Falta el archivo {f} en {CARPETA}")

def ip_local():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()
    def log_message(self, *a): pass          # consola limpia

class Servidor(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

url = f"http://{ip_local()}:{PUERTO}"
print("\n  DetectaGrietas funcionando")
print(f"  En este computador : http://localhost:{PUERTO}")
print(f"  En el celular      : {url}   (misma red WiFi)\n")
try:                                          # QR opcional: pip install qrcode
    import qrcode
    qr = qrcode.QRCode(border=1); qr.add_data(url); qr.print_ascii(invert=True)
except ImportError:
    print("  (Opcional: 'pip install qrcode' para ver un QR aqui)")
print("\n  Ctrl+C para detener.\n")

with Servidor(("", PUERTO), functools.partial(Handler, directory=CARPETA)) as srv:
    try: srv.serve_forever()
    except KeyboardInterrupt: print("Servidor detenido.")
