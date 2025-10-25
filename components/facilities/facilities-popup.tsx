"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { NearbyFacilities } from "./nearby-facilities"
import { MapPin } from "lucide-react"

export function FacilitiesPopup() {
  const [open, setOpen] = useState(false)

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          className="w-full justify-start gap-2 bg-sidebar-accent/30 border-border text-foreground/90 hover:bg-sidebar-accent/50 hover:text-foreground"
        >
          <MapPin className="h-4 w-4 text-primary" /> <span className="font-medium">Nearby Facilities</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl bg-background/95 backdrop-blur border-border text-foreground max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-lg">
            <MapPin className="h-5 w-5 text-primary" /> Clinics & Hospitals Near You
          </DialogTitle>
        </DialogHeader>
        <div className="pb-4">
          <NearbyFacilities active={open} />
        </div>
      </DialogContent>
    </Dialog>
  )
}
