
(cl:in-package :asdf)

(defsystem "lidar-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "PolygonArray" :depends-on ("_package_PolygonArray"))
    (:file "_package_PolygonArray" :depends-on ("_package"))
  ))