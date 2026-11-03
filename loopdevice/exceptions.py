class DeviceError(Exception):
    pass

class HandshakeError(DeviceError):
    pass

class ConnectionError(DeviceError):
    pass
